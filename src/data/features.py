"""Utterance-level view over the sharded Whisper feature files."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AudioFeatureDataset(Dataset):
    """Utterance-level view over the sharded (frames, d_model) feature files.

    The split is ~103 GB on disk, so only the requested slice is read into RAM, with one
    positioned read at its byte offset; the OS page cache keeps hot shards resident. On
    GPFS that read is ~1.75x faster than faulting the same slice in through a memory map
    (Snellius login node, 16 processes over random train.960 utterances: 68-85 utt/s
    against 39-47).
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
        # shard -> (path, data offset, dtype, d_model), read from the .npy header on first use
        self._shards: dict[int, tuple[Path, int, np.dtype, int]] = {}

    def __len__(self) -> int:
        """Number of utterances across every shard."""
        return len(self.utts)

    @property
    def texts(self) -> list[str]:
        """Every transcript, in dataset order."""
        return [u[3] for u in self.utts]

    def _read(self, shard_idx: int, offset: int, length: int) -> np.ndarray:
        """Frames [offset, offset + length) of one shard, as an array that owns its memory."""
        if shard_idx not in self._shards:
            path = self.index_paths[shard_idx].with_suffix(".npy")
            header = np.load(path, mmap_mode="r")  # parses the header; no data is read
            self._shards[shard_idx] = (path, header.offset, header.dtype, header.shape[1])
        path, start, dtype, d_model = self._shards[shard_idx]
        flat = np.fromfile(
            path, dtype, count=length * d_model, offset=start + offset * d_model * dtype.itemsize
        )
        return flat.reshape(length, d_model)

    def __getitem__(self, idx: int) -> dict:
        """One utterance: its (frames, d_model) latents and its raw transcript.

        Whisper frames are 20 ms; stride 4 makes them 80 ms, still well under a phoneme,
        and it cuts the cross-attention keys 4x. The pool accumulates in float32 so the
        mean does not sum in float16, then returns to the stored dtype: the result leaves
        this process through shared memory, and the training step casts it up on the GPU
        anyway, so widening it here would double the per-batch transfer for nothing.
        """
        shard_idx, offset, length, text = self.utts[idx]
        audio = torch.from_numpy(self._read(shard_idx, offset, length))
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
