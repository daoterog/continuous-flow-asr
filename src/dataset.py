"""Dataset, compact vocabulary and the DataModule over the precomputed Whisper latents.

Splits are named the way the download and precompute scripts name them: a role (train,
val, test) is a set of `configs` x `splits` pairs, each resolving to one feature
directory under `data/features/<model>/<config>/<split>/`. That keeps
``--train-configs clean other --train-splits train.100 train.360`` meaning the same thing
here as it does for ``scripts/precompute_whisper.py``.
"""

import json
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path

from src.constants import CONFIGS, SPLITS, feature_dir

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Subset
from transformers import WhisperTokenizerFast

ROLES = ("train", "val", "test")


class AudioFeatureDataset(Dataset):
    """Utterance-level view over the sharded (frames, d_model) feature files.

    The split is ~103 GB on disk, so the .npy shards stay memory-mapped and only the
    requested slice is copied into RAM; the OS page cache keeps hot shards resident.
    The .json indexes are small, so they are read once up front and flattened into a
    single utterance table (shards are not all the same size, so idx can't be split
    arithmetically).
    """

    def __init__(self, audio_feature_dir: Path, audio_stride: int = 1) -> None:
        """Read every shard index under `audio_feature_dir` into one utterance table."""
        self.audio_feature_dir = Path(audio_feature_dir)
        self.audio_stride = audio_stride
        self.index_paths = sorted(self.audio_feature_dir.glob("*.json"))
        if not self.index_paths:
            raise FileNotFoundError(f"no feature shards under {self.audio_feature_dir}")

        self.utts: list[tuple[int, int, int, str]] = []  # (shard, offset, length, text)
        d_models = set()
        for shard_idx, index_path in enumerate(self.index_paths):
            index = json.loads(index_path.read_text())
            # Whisper's decoder cross-attends over all 1500 frames, so the cache must be
            # the untrimmed one; a trimmed cache silently gives unusable short latents.
            # Checked per shard: a directory can be half-rewritten by a partial rerun.
            if index["trimmed"]:
                raise ValueError(
                    f"{index_path} holds trimmed latents; re-run the precompute with "
                    f"NO_TRIM=1 for anything fed back through Whisper's decoder"
                )
            d_models.add(index["d_model"])
            self.utts += [
                (shard_idx, u["offset"], u["length"], u["text"]) for u in index["utts"]
            ]
        if len(d_models) != 1:
            raise ValueError(
                f"shards under {self.audio_feature_dir} disagree on d_model: {d_models}"
            )
        self.d_model = d_models.pop()
        self._shards: dict[int, np.ndarray] = {}  # lazy, so each worker gets its own mmap

    def __len__(self) -> int:
        """Number of utterances across every shard."""
        return len(self.utts)

    @property
    def texts(self) -> list[str]:
        """Every transcript, in dataset order."""
        return [u[3] for u in self.utts]

    def _feats(self, shard_idx: int) -> np.ndarray:
        if shard_idx not in self._shards:
            path = self.index_paths[shard_idx].with_suffix(".npy")
            self._shards[shard_idx] = np.load(path, mmap_mode="r")
        return self._shards[shard_idx]

    def __getitem__(self, idx: int) -> dict:
        """One utterance: its (frames, d_model) latents and its raw transcript.

        Whisper frames are 20 ms; stride 4 makes them 80 ms, still well under a phoneme,
        and it cuts the cross-attention keys 4x. The pool accumulates in float32 so the
        mean does not sum in float16, then returns to the stored dtype: the result leaves
        this process through shared memory, and the training step casts it up on the GPU
        anyway, so widening it here would double the per-batch transfer for nothing.
        """
        shard_idx, offset, length, text = self.utts[idx]
        feats = self._feats(shard_idx)[offset : offset + length]
        # copy off the read-only mmap so the tensor owns its memory
        audio = torch.from_numpy(np.array(feats))
        if self.audio_stride > 1:
            frames, d_model = audio.shape
            keep = frames // self.audio_stride * self.audio_stride
            audio = (
                audio[:keep]
                .view(keep // self.audio_stride, self.audio_stride, d_model)
                .float()
                .mean(dim=1)
                .to(audio.dtype)
            )
        return {"audio": audio, "text": text}


class VocabMap:
    """Lossless remap of the Whisper ids that occur in the corpus onto [0, size).

    LibriSpeech is uppercase and unpunctuated, so it touches a small fraction of
    Whisper's 51,866 tokens. The rest are slots the simplex has to carry and the model can
    never use. Dropping them is exact — every id round-trips — and it is what makes a
    (B, L, V) field affordable on a laptop.

    Build it from the union of every split that will be encoded, or `encode` will trip on
    a validation token that never appeared in training.
    """

    def __init__(self, texts: Sequence[str], tokenizer: WhisperTokenizerFast) -> None:
        """Measure which Whisper ids `texts` uses and build both directions of the map."""
        used = sorted({i for ids in tokenizer(list(texts)).input_ids for i in ids})
        self.to_full = torch.tensor(used, dtype=torch.long)
        rev = torch.full((len(tokenizer),), -1, dtype=torch.long)
        rev[self.to_full] = torch.arange(len(used))
        self.to_compact = rev
        self.size = len(used)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Whisper ids -> compact ids."""
        out = self.to_compact.to(ids.device)[ids]
        if not bool((out >= 0).all()):
            raise ValueError(
                "token outside the compact vocabulary — build the VocabMap from the "
                "union of every split you encode, not from train alone"
            )
        return out

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        """Compact ids -> Whisper ids."""
        return self.to_full.to(ids.device)[ids]


def collate(
    batch: list[dict],
    tokenizer: WhisperTokenizerFast,
    canvas_len: int,
    vocab_map: VocabMap | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize onto the fixed canvas and stack the (frames, d_model) latents.

    The flow denoises a whole canvas_len-wide field at once, so the target width has to
    be identical every batch — at sampling time there is no reference transcript to size
    it from. Padding is <|endoftext|> (Whisper's pad_token_id) and it is a real target:
    the model learns to emit it, which is how a non-autoregressive sample knows where to
    stop. That is also why nothing here builds an attention mask — every canvas position
    is predicted, so there is nothing to exclude.
    """
    enc = tokenizer(
        [b["text"] for b in batch],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=canvas_len,
    )
    tokens = enc.input_ids
    if vocab_map is not None:
        tokens = vocab_map.encode(tokens)

    # every clip is the same length now that the cache is untrimmed, and the dataset has
    # already pooled along time, so a plain stack does it
    return {"audio": torch.stack([b["audio"] for b in batch]), "tokens": tokens}


def expand_spec(configs: Iterable[str], splits: Iterable[str]) -> list[tuple[str, str]]:
    """Cross `configs` with `splits`, as the download and precompute scripts do.

    Validated against the whitelists so a typo fails here rather than as an empty role
    after the feature directories are probed.
    """
    configs, splits = list(configs), list(splits)
    for name, got, allowed in (("config", configs, CONFIGS), ("split", splits, SPLITS)):
        bad = [x for x in got if x not in allowed]
        if bad:
            raise ValueError(f"unknown {name}(s) {bad}; expected from {allowed}")
    return [(c, s) for c in configs for s in splits]


def resolve_role(
    data_root: Path, model: str, configs: Iterable[str], splits: Iterable[str], role: str
) -> list[Path]:
    """Feature directories for one role, skipping (config, split) pairs that do not exist.

    Skipping rather than rejecting matches the scripts: LibriSpeech has train.clean.100,
    train.clean.360 and train.other.500, so `--configs clean other --splits train.100
    train.360 train.500` is the natural way to ask for all three, and two of its six pairs
    are not real. A role that resolves to nothing is still an error.
    """
    pairs = expand_spec(configs, splits)
    found = [
        d
        for c, s in pairs
        if (d := feature_dir(data_root, model, c, s)).is_dir() and any(d.glob("*.json"))
    ]
    if not found:
        wanted = ", ".join(f"{c}/{s}" for c, s in pairs)
        raise FileNotFoundError(
            f"no feature directories for role '{role}' ({wanted}) under "
            f"{data_root / 'features' / model.split('/')[-1]}.\nPrecompute them with:\n"
            f"  CONFIGS=\"{' '.join(sorted({c for c, _ in pairs}))}\" "
            f"SPLITS=\"{' '.join(sorted({s for _, s in pairs}))}\" NO_TRIM=1 "
            f"./jobs/submit/whisper.sh"
        )
    return found


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn: Callable,
    num_workers: int = 4,
    pin_memory: bool = False,
    steps_per_epoch: int | None = None,
) -> DataLoader:
    """One DataLoader, with the settings shared by every split.

    steps_per_epoch decouples the epoch from the dataset size by sampling with
    replacement. On a small subset an epoch is otherwise only a handful of batches, and
    the fixed per-epoch cost (~100 ms of loader restart and hooks) comes to dominate:
    25,600 steps over 64 utterances at batch 16 is 6,400 epoch boundaries. Setting it
    makes an epoch as long as you like without changing what the model sees.
    """
    sampler = None
    if steps_per_epoch:
        sampler = RandomSampler(
            ds, replacement=True, num_samples=steps_per_epoch * batch_size
        )
        shuffle = False  # mutually exclusive with an explicit sampler
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,  # keeps the step count exact for the scheduler
    )


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
        audio_stride: int = 1,
        compact_vocab: bool = True,
        subset_n: dict | None = None,
        num_workers: int = 4,
        seed: int = 0,
        pin_memory: bool = False,
        steps_per_epoch: int | None = None,
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
        self.audio_stride = audio_stride
        self.compact_vocab = compact_vocab
        self.subset_n = subset_n or {}
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.steps_per_epoch = steps_per_epoch

        self.datasets: dict[str, Subset] = {}
        self.dirs: dict[str, list[Path]] = {}
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

    def _loader(self, role: str, shuffle: bool) -> DataLoader:
        if not self._is_set_up:
            self.setup()
        return make_loader(
            self.datasets[role],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Only the training loader: val and test must cover their split exactly once.
            steps_per_epoch=self.steps_per_epoch if role == "train" else None,
        )

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
            lines.append(f"  {role:5s} {kept:>7,} of {total:>7,} utts   [{dirs}]")
        return "\n".join(lines)
