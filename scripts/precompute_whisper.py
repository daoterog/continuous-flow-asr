"""Precompute and cache frozen Whisper encoder features for LibriSpeech.

The encoder's weights never change during training, so running it inside the training
loop redoes identical work every epoch. Caching its output once turns every later epoch
into a disk read.

Whisper's feature extractor pads every clip to a 30 s window (480,000 samples ->
3000 mel frames -> 1500 encoder frames) regardless of the real duration, and LibriSpeech
averages 12.7 s. By default each output is therefore sliced back to
``ceil(valid_mel_frames / 2)``: the values are identical, only the padding rows are
dropped, and train.clean.100 shrinks from 110 GB to 46 GB. ``--no-trim`` keeps the raw
fixed-1500 tensors.

Output is one memory-mappable ``.npy`` plus a JSON index per input parquet shard —
14 file pairs for train.100 rather than 28,539 small files, which matters on Lustre.
Utterance ``i`` is ``feats[offset[i] : offset[i] + length[i]]``.

Work parallelises across SLURM array tasks at the shard level: task ``k`` of ``N`` takes
``shards[k::N]``, so the maximum useful ``--num-shards`` is the number of shards
selected.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.download_librispeech import (
    CONFIGS,
    LOCAL_DATA_ROOT,
    REPO_ID,
    SNELLIUS_DATA_ROOT,
    SPLITS,
)

if TYPE_CHECKING:  # runtime imports are deferred; torch is slow and only needed on use
    import torch
    from torch.utils.data import DataLoader
    from transformers import WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

DEFAULT_MODEL = "openai/whisper-large-v3"

# Whisper pads to a fixed 30 s window; these follow from its preprocessor config and
# are asserted against the loaded extractor at runtime rather than trusted blindly.
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
# The encoder's second conv layer has stride 2, so it emits one frame per two mel frames.
MEL_TO_ENCODER_STRIDE = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    p = argparse.ArgumentParser(
        description="Precompute Whisper encoder features for LibriSpeech.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--data-root", type=Path, default=None, help="overrides --env")

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=["clean"])
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=["train.100"])

    p.add_argument(
        "--num-shards",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
        help="number of array tasks sharing the work",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="index of this task, in [0, --num-shards)",
    )
    p.add_argument("--batch-size", type=int, default=8, help="encoder forward batch")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count(),
        help="DataLoader workers decoding audio and computing mels",
    )

    p.add_argument(
        "--no-trim",
        action="store_true",
        help="store the raw fixed 1500 frames per utterance (2.4x the disk)",
    )
    p.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="encoder and storage precision; float32 doubles disk and halves speed",
    )
    p.add_argument("--limit", type=int, default=None, help="stop after N utts per shard")
    p.add_argument("--overwrite", action="store_true", help="recompute finished shards")
    p.add_argument("--failure-log", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--verify", action="store_true", help="check outputs; compute nothing")

    args = p.parse_args(argv)

    # An empty model would write features to features//<config>/<split>, outside the
    # per-checkpoint namespacing everything else relies on.
    if not args.model:
        p.error("--model must not be empty")
    if args.num_shards < 1:
        p.error(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        p.error(f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}")
    if args.batch_size < 1:
        p.error(f"--batch-size must be >= 1, got {args.batch_size}")

    if args.data_root is None:
        args.data_root = SNELLIUS_DATA_ROOT if args.env == "snellius" else LOCAL_DATA_ROOT
    args.data_root = args.data_root.expanduser().resolve()

    if args.failure_log is None:
        args.failure_log = (
            args.data_root / "logs" / f"whisper_failures_{args.shard_id}.json"
        )

    return args


def snapshot_dir(data_root: Path) -> Path:
    """Locate the cached LibriSpeech snapshot under the data root."""
    snapshots = data_root / "hub" / f"datasets--{REPO_ID.replace('/', '--')}" / "snapshots"
    available = sorted(p for p in snapshots.glob("*") if p.is_dir())
    if not available:
        raise FileNotFoundError(
            f"no LibriSpeech cache under {snapshots} — run scripts.download_librispeech first"
        )
    if len(available) > 1:
        raise ValueError(f"{len(available)} revisions cached under {snapshots}, expected one")
    return available[0]


def select_shards(
    snapshot: Path, configs: list[str], splits: list[str]
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Return (sorted parquet shards, skipped (config, split) pairs)."""
    shards, skipped = [], []
    for config in configs:
        for split in splits:
            split_dir = snapshot / config / split
            found = sorted(split_dir.glob("*.parquet"))
            if found:
                shards.extend(found)
            else:
                skipped.append((config, split))
    return sorted(shards), skipped


def output_paths(data_root: Path, model: str, shard: Path) -> tuple[Path, Path]:
    """Return the (.npy, .json) output pair mirroring the input shard's location."""
    # .../snapshots/<sha>/clean/train.100/0000.parquet -> features/<model>/clean/train.100/0000
    out_dir = (
        data_root / "features" / model.split("/")[-1]
        / shard.parent.parent.name / shard.parent.name
    )
    return out_dir / f"{shard.stem}.npy", out_dir / f"{shard.stem}.json"


def count_utterances(shards: list[Path], limit: int | None) -> int:
    """Total rows across ``shards``, read from parquet footers without loading data."""
    import pyarrow.parquet as pq

    total = 0
    for shard in shards:
        rows = pq.ParquetFile(shard).metadata.num_rows
        total += min(rows, limit) if limit is not None else rows
    return total


def is_complete(shard: Path, npy: Path, index: Path) -> bool:
    """True if the outputs exist, are self-consistent, and cover the whole input shard.

    The utterance count is checked against the input parquet, not just against the array:
    a ``--limit`` run produces a perfectly self-consistent index over a handful of rows,
    which would otherwise be mistaken for a finished shard and skipped on resume.
    """
    if not (npy.is_file() and index.is_file()):
        return False
    try:
        import numpy as np
        import pyarrow.parquet as pq

        meta = json.loads(index.read_text())
        rows = np.load(npy, mmap_mode="r").shape[0]
        expected_utts = pq.ParquetFile(shard).metadata.num_rows
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return rows == sum(u["length"] for u in meta["utts"]) and len(meta["utts"]) == expected_utts


def build_loader(
    shard: Path, extractor: "WhisperFeatureExtractor", args: argparse.Namespace
) -> "DataLoader":
    """Build a DataLoader yielding (mel, valid_mel_frames, id, text) batches.

    Audio decode plus mel extraction is CPU work and is the usual bottleneck here, so it
    runs across ``--workers`` processes to keep the GPU fed.
    """
    import torch
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=str(shard), split="train")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, ds.num_rows)))

    def encode(batch: dict) -> dict:
        audio = batch["audio"]
        arr = audio["array"]
        # Captured before the extractor pads to 30 s — unrecoverable afterwards.
        valid = min(len(arr) // HOP_LENGTH, MAX_MEL_FRAMES)
        mel = extractor(
            arr,
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]
        return {"mel": mel, "valid": valid, "id": batch["id"], "text": batch["text"]}

    class _Wrapped(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return ds.num_rows

        def __getitem__(self, i: int) -> dict:
            return encode(ds[i])

    def collate(items: list[dict]) -> dict:
        return {
            "mel": torch.stack([i["mel"] for i in items]),
            "valid": [i["valid"] for i in items],
            "id": [i["id"] for i in items],
            "text": [i["text"] for i in items],
        }

    return torch.utils.data.DataLoader(
        _Wrapped(),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
        shuffle=False,
    )


def process_shard(
    shard: Path,
    encoder: "WhisperEncoder",
    extractor: "WhisperFeatureExtractor",
    device: "torch.device",
    args: argparse.Namespace,
) -> int:
    """Encode one parquet shard and write its .npy/.json pair. Returns utterances done.

    Features are streamed straight into an on-disk memmap rather than accumulated in a
    list. Holding a whole untrimmed shard in RAM and then concatenating it needs roughly
    2 x 9.4 GiB for clean/test, which the OOM killer ends on a machine with less; writing
    through the memmap keeps peak RAM at one batch.
    """
    import numpy as np
    import torch

    npy, index = output_paths(args.data_root, args.model, shard)
    npy.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(args.dtype)
    d_model = int(encoder.config.d_model)
    max_frames = MAX_MEL_FRAMES // MEL_TO_ENCODER_STRIDE
    capacity = count_utterances([shard], args.limit) * max_frames

    # Write to .tmp then rename: a killed job must not leave a half-written shard that
    # looks complete to the resume check.
    tmp = npy.with_suffix(".npy.tmp")
    store = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=dtype, shape=(capacity, d_model)
    )

    utts: list[dict] = []
    offset = 0
    try:
        for batch in build_loader(shard, extractor, args):
            mel = batch["mel"].to(device, dtype=encoder.dtype)
            with torch.no_grad():
                out = encoder(mel).last_hidden_state
            # Cast on the GPU so only the stored precision crosses the bus.
            out = out.to(getattr(torch, dtype.name)).cpu().numpy()

            for row, valid, utt_id, text in zip(
                out, batch["valid"], batch["id"], batch["text"], strict=True
            ):
                length = (
                    row.shape[0]
                    if args.no_trim
                    else min(math.ceil(valid / MEL_TO_ENCODER_STRIDE), row.shape[0])
                )
                store[offset : offset + length] = row[:length]
                utts.append({"id": utt_id, "offset": offset, "length": length, "text": text})
                offset += length
        store.flush()
    finally:
        del store  # release the mapping before the file is renamed or rewritten

    if offset == capacity:
        # Untrimmed: the upper bound was exact, so the file is already correct.
        os.replace(tmp, npy)
    else:
        # Trimmed: copy the used prefix into a right-sized file, a chunk at a time so
        # peak RAM stays bounded regardless of shard size.
        final = npy.with_suffix(".npy.tmp2")
        src = np.load(tmp, mmap_mode="r")
        dst = np.lib.format.open_memmap(
            final, mode="w+", dtype=dtype, shape=(offset, d_model)
        )
        for start in range(0, offset, 1 << 16):
            end = min(start + (1 << 16), offset)
            dst[start:end] = src[start:end]
        dst.flush()
        del src, dst
        os.replace(final, npy)
        tmp.unlink()

    index.write_text(
        json.dumps(
            {
                "model": args.model,
                "dtype": str(dtype),
                "d_model": d_model,
                "trimmed": not args.no_trim,
                "source": shard.name,
                "utts": utts,
            },
            indent=2,
        )
        + "\n"
    )
    return len(utts)


def load_encoder(
    model: str, dtype: str
) -> tuple["WhisperEncoder", "WhisperFeatureExtractor", "torch.device"]:
    """Load the Whisper encoder and feature extractor onto the best available device."""
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # One dtype governs both the encode and the storage: casting fp16 output up to fp32
    # is lossless but adds no information, so the two are only ever meaningful together.
    # The exception is CPU, where fp16 is unsupported for these ops.
    torch_dtype = torch.float32 if device.type == "cpu" else getattr(torch, dtype)

    extractor = WhisperFeatureExtractor.from_pretrained(model)
    if extractor.hop_length != HOP_LENGTH:
        raise ValueError(f"expected hop_length {HOP_LENGTH}, got {extractor.hop_length}")

    full = WhisperModel.from_pretrained(model, dtype=torch_dtype)
    encoder = full.get_encoder().to(device).eval()
    del full.decoder  # the decoder is loaded by from_pretrained but never used here
    return encoder, extractor, device


def main(argv: list[str] | None = None) -> int:
    """Run the precompute, returning a process exit code."""
    args = parse_args(argv)
    # Set rather than setdefault: the Whisper checkpoint is ~3 GB and an inherited
    # HF_HOME pointing at a quota'd home directory would otherwise win.
    os.environ["HF_HOME"] = str(args.data_root)

    snapshot = snapshot_dir(args.data_root)
    shards, skipped = select_shards(snapshot, args.configs, args.splits)
    for config, split in skipped:
        print(f"warning: {config}/{split} not in the local cache, skipping", file=sys.stderr)
    if not shards:
        print("error: no shards selected", file=sys.stderr)
        return 1

    mine = shards[args.shard_id :: args.num_shards]
    def done(shard: Path) -> bool:
        return is_complete(shard, *output_paths(args.data_root, args.model, shard))

    todo = [s for s in mine if args.overwrite or not done(s)]

    print(f"data root   {args.data_root}")
    print(f"model       {args.model}")
    print(f"configs     {' '.join(args.configs)}")
    print(f"splits      {' '.join(args.splits)}")
    print(
        f"shard       {args.shard_id}/{args.num_shards} "
        f"-> {len(mine)} of {len(shards)} shard(s)"
    )
    print(f"pending     {len(todo)} (skipping {len(mine) - len(todo)} already complete)")
    # Machine-readable; submit/whisper.sh reads this to size the sbatch --time.
    print(f"utterances  {count_utterances(mine, args.limit)}")
    print(f"trim        {'off' if args.no_trim else 'on'}")
    print(f"dtype       {args.dtype}")
    print(f"batch/work  {args.batch_size} / {args.workers}")

    if args.dry_run:
        for s in mine:
            print(s)
        return 0

    if args.verify:
        incomplete = [s for s in mine if not done(s)]
        if incomplete:
            print(f"MISSING {len(incomplete)} of {len(mine)} shard(s):", file=sys.stderr)
            for s in incomplete:
                print(f"  {s.name}", file=sys.stderr)
            return 1
        print(f"OK: all {len(mine)} shard(s) present")
        return 0

    if not todo:
        print("nothing to do")
        return 0

    encoder, extractor, device = load_encoder(args.model, args.dtype)
    print(f"device      {device} ({encoder.dtype})")

    failures: list[dict] = []
    for i, shard in enumerate(todo, 1):
        rel = f"{shard.parent.parent.name}/{shard.parent.name}/{shard.name}"
        print(f"[{i}/{len(todo)}] {rel}", flush=True)
        started = time.perf_counter()
        try:
            n = process_shard(shard, encoder, extractor, device, args)
        except Exception as exc:  # one bad shard must not lose the whole task
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            failures.append({"shard": rel, "error": repr(exc)})
        else:
            # flush=True throughout: stdout is block-buffered to the SLURM output file,
            # so without it a long encode shows nothing and a finished shard is not
            # visibly finished. The rate is what sizes the next job's walltime.
            elapsed = time.perf_counter() - started
            print(
                f"  {n} utterance(s) in {elapsed:.0f}s ({n / max(elapsed, 1e-9):.1f} utts/s)",
                flush=True,
            )

    if failures:
        args.failure_log.parent.mkdir(parents=True, exist_ok=True)
        args.failure_log.write_text(
            json.dumps(
                {"shard_id": args.shard_id, "num_shards": args.num_shards, "failures": failures},
                indent=2,
            )
        )
        print(
            f"FAILED: {len(failures)} of {len(todo)} shard(s), logged to {args.failure_log}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        f"OK: {len(todo)} shard(s) written under {args.data_root / 'features'}", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
