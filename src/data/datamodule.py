"""The LightningDataModule: one tokenization pass, the shared vocabulary, the loaders."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import WhisperTokenizerFast

from .collate import CudaPrefetcher, collate, make_loader
from .features import AudioFeatureDataset
from .spec import resolve_role
from .vocab import VocabMap

ROLES = ("train", "val", "test")


class AudioFeatureDataModule(pl.LightningDataModule):
    """Train/val/test over the precomputed latents, one role per set of (config, split).

    Sizing the models needs `vocab_size`, `pad_id` and `d_model`, and all three are
    measured from the data — so `setup()` has to run before the LightningModule is built.
    `prepare()` does exactly that and is safe to call directly from a script.
    """

    def __init__(
        self,
        data_root: Path,
        model: str,
        splits: dict,
        tokenizer: WhisperTokenizerFast,
        canvas_len: int,
        batch_size: int,
        eval_batch_size: int | None = None,
        audio_stride: int = 1,
        compact_vocab: bool = True,
        subset_n: dict | None = None,
        num_workers: int = 4,
        seed: int = 0,
        pin_memory: bool = False,
        steps_per_epoch: int | None = None,
        gpu_prefetch: bool = False,
        prefetch_factor: int | None = None,
        eval_num_workers: int | None = None,
    ) -> None:
        """Record the configuration; nothing is read off disk until `setup()`.

        splits: {"train": {"configs": [...], "splits": [...]}, "val": ..., "test": ...}
        subset_n: optional {"train": 2000, ...} cap per role, applied after the canvas
        length filter and seeded off `seed` so the subsample is reproducible.
        """
        super().__init__()
        missing = [r for r in ROLES if r not in splits]
        if missing:
            raise KeyError(f"splits is missing role(s) {missing}; it needs {list(ROLES)}")
        self.data_root = Path(data_root)
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer
        self.canvas_len = canvas_len
        self.batch_size = batch_size
        self.eval_batch_size = int(eval_batch_size or batch_size)
        self.audio_stride = audio_stride
        self.compact_vocab = compact_vocab
        self.subset_n = subset_n or {}
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.steps_per_epoch = steps_per_epoch
        # Copy the next batch to the GPU while the current one trains (CudaPrefetcher).
        self.gpu_prefetch = gpu_prefetch and pin_memory and torch.cuda.is_available()
        self.prefetch_factor = prefetch_factor
        # Val and test run every few epochs and keep their workers alive in between, so they
        # need far fewer than training; None gives them training's count.
        self.eval_num_workers = num_workers if eval_num_workers is None else eval_num_workers

        self.datasets: dict[str, Subset] = {}
        self.dirs: dict[str, list[Path]] = {}
        self.mean_len: dict[str, float] = {}
        self.vocab_map: VocabMap | None = None
        self.vocab_size: int | None = None
        self.pad_id: int | None = None
        self.d_model: int | None = None
        self._is_set_up = False

    def _load_role(self, role: str) -> tuple[ConcatDataset, list[Path]]:
        spec = self.splits[role]
        dirs = resolve_role(
            self.data_root, self.model, spec["configs"], spec["splits"], role
        )
        parts = [AudioFeatureDataset(d, self.audio_stride) for d in dirs]
        d_models = {p.d_model for p in parts}
        if len(d_models) != 1:
            raise ValueError(f"role '{role}' mixes feature widths {d_models}")
        return ConcatDataset(parts), dirs

    def setup(self, stage: str | None = None) -> None:
        """Load every role, build the vocabulary, filter to the canvas and subsample."""
        if self._is_set_up:
            return

        raw, texts = {}, {}
        for role in ROLES:
            raw[role], self.dirs[role] = self._load_role(role)
            texts[role] = [t for part in raw[role].datasets for t in part.texts]
        self.d_model = raw["train"].datasets[0].d_model

        # Union across every role: a VocabMap built from train alone trips on the first
        # validation token that never appeared in training.
        if self.compact_vocab:
            self.vocab_map = VocabMap(
                [t for role in ROLES for t in texts[role]], self.tokenizer
            )
            self.vocab_size = self.vocab_map.size
            self.pad_id = int(self.vocab_map.to_compact[self.tokenizer.pad_token_id])
        else:
            self.vocab_map = None
            self.vocab_size = len(self.tokenizer)
            self.pad_id = self.tokenizer.pad_token_id

        # One tokenization pass per role serves the length filter, so no role ever
        # contains an utterance the canvas would truncate.
        for role in ROLES:
            lengths = np.array([len(x) for x in self.tokenizer(texts[role]).input_ids])
            eligible = np.flatnonzero(lengths <= self.canvas_len)
            cap = self.subset_n.get(role)
            if cap is not None and cap < len(eligible):
                eligible = np.random.default_rng(self.seed).choice(
                    eligible, cap, replace=False
                )
            self.datasets[role] = Subset(raw[role], sorted(eligible.tolist()))
            # Kept for the summary: it is the ratio that says how much of the canvas the
            # token-level metrics are really about, the rest being padding they exclude.
            self.mean_len[role] = float(lengths[eligible].mean()) if len(eligible) else 0.0

        self._is_set_up = True

    def prepare(self) -> "AudioFeatureDataModule":
        """`setup()` that returns self, for `dm = AudioFeatureDataModule(...).prepare()`."""
        self.setup()
        return self

    def texts(self, role: str) -> list[str]:
        """The transcripts of one role, after filtering and subsampling."""
        if not self._is_set_up:
            self.setup()
        subset = self.datasets[role]
        every = [t for part in subset.dataset.datasets for t in part.texts]
        return [every[i] for i in subset.indices]

    def sample_tokens(self, role: str, n: int, seed: int = 0) -> torch.Tensor:
        """(n, canvas_len) canvas tokens for `n` random utterances of `role`.

        Tokenizes the transcripts directly instead of going through the DataLoader: the
        corruption sweep needs only the targets, and collating the latents alongside them
        would pull hundreds of MB off the mmap for nothing.
        """
        texts = self.texts(role)
        n = min(n, len(texts))
        picks = np.random.default_rng(seed).choice(len(texts), n, replace=False)
        enc = self.tokenizer(
            [texts[i] for i in picks],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.canvas_len,
        )
        tokens = enc.input_ids
        return self.vocab_map.encode(tokens) if self.vocab_map is not None else tokens

    @property
    def collate_fn(self) -> Callable:
        """The collate bound to this module's tokenizer, canvas and vocabulary."""
        return partial(
            collate,
            tokenizer=self.tokenizer,
            canvas_len=self.canvas_len,
            vocab_map=self.vocab_map,
        )

    def _loader(self, role: str, shuffle: bool) -> DataLoader | CudaPrefetcher:
        if not self._is_set_up:
            self.setup()
        loader = make_loader(
            self.datasets[role],
            batch_size=self.batch_size if role == "train" else self.eval_batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if role == "train" else self.eval_num_workers,
            pin_memory=self.pin_memory,
            # Only the training loader: val and test must cover their split exactly once.
            steps_per_epoch=self.steps_per_epoch if role == "train" else None,
            prefetch_factor=self.prefetch_factor,
        )
        return CudaPrefetcher(loader) if self.gpu_prefetch else loader

    def train_dataloader(self) -> DataLoader:
        """Shuffled, with drop_last so the LR schedule's step count is exact."""
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Unshuffled; every batch is kept."""
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Unshuffled. Never touched by `fit` — only by an explicit `trainer.test()`."""
        return self._loader("test", shuffle=False)

    def summary(self) -> str:
        """A few lines naming the resolved directories and the measured sizes."""
        if not self._is_set_up:
            self.setup()
        lines = [
            f"vocab {len(self.tokenizer):,} -> {self.vocab_size:,}   "
            f"canvas {self.canvas_len}   audio stride {self.audio_stride}   "
            f"batch {self.batch_size}   d_model {self.d_model}   pad_id {self.pad_id}"
        ]
        for role in ROLES:
            dirs = ", ".join(f"{d.parent.name}/{d.name}" for d in self.dirs[role])
            kept, total = len(self.datasets[role]), len(self.datasets[role].dataset)
            # The padding share is printed because it is what separates a token-level
            # metric from a word-level one: at 80% padding a model with every real token
            # wrong still scores 0.2 if pad positions are counted. The metrics exclude
            # them, so this line says how much of the canvas each number is actually about.
            mean_len = self.mean_len[role]
            lines.append(
                f"  {role:5s} {kept:>7,} of {total:>7,} utts   "
                f"mean {mean_len:5.1f} tok ({100 * (1 - mean_len / self.canvas_len):2.0f}% "
                f"of canvas is pad)   [{dirs}]"
            )
        return "\n".join(lines)
