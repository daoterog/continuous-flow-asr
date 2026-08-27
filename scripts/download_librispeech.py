"""Download LibriSpeech parquet shards into an HF hub cache.

Download-only: fetches the raw parquet files without the Arrow re-encode that
``load_dataset`` performs, so a staging node spends its time on I/O rather than CPU.

Work is parallelised across SLURM array tasks at the *shard* level.  Shards are
near-uniform (~500 MB) while splits are not (0.35 GB -> 31.8 GB), so slicing shards
keeps every task's slice the same size.  Task ``k`` of ``N`` takes ``files[k::N]``;
slices are disjoint, which is what makes concurrent writes into one cache safe.

Note the ``all`` config is deliberately unreachable from ``--configs``: it is the same
audio as ``clean`` + ``other`` under merged split names, so including it would download
the whole corpus a second time.

Read the result back with::

    HF_HOME=<data-root> load_dataset("openslr/librispeech_asr", "clean", split="train.100")
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # runtime import is deferred until after setup_hf_env()
    from huggingface_hub import HfApi

REPO_ID = "openslr/librispeech_asr"

# The dataset card carries the config/split definitions; without it `load_dataset`
# cannot resolve a config name and fails outright in offline mode.  Kept out of the
# shard slicing so every task fetches it — it is tiny, and hf_hub takes a per-file
# lock, so concurrent fetches of the same file are safe and idempotent.
METADATA_FILES = ["README.md"]

# Splits are per-config; `--configs` x `--splits` is a cross product and pairs that do
# not exist are skipped rather than rejected, so one `--splits` list works for both.
CONFIGS = ["clean", "other"]
SPLITS = ["train.100", "train.360", "train.500", "validation", "test"]

SNELLIUS_DATA_ROOT = Path("/scratch-shared/cont-flow-asr/data")
LOCAL_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    p = argparse.ArgumentParser(
        description="Download LibriSpeech parquet shards into an HF hub cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--env",
        choices=["local", "snellius"],
        default="local",
        help=f"where this is running; picks the data root "
        f"(local -> {LOCAL_DATA_ROOT}, snellius -> {SNELLIUS_DATA_ROOT})",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="explicit data root; overrides --env",
    )

    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)

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
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count(),
        help="concurrent downloads within this task",
    )

    p.add_argument("--revision", default="main", help="resolved to a sha before use")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--no-fast-transfer",
        action="store_true",
        help="disable Xet high-performance mode (lower memory, slower)",
    )
    p.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="default: <data-root>/logs/failures_<shard-id>.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan and the shard's file list, then exit",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="check the shard's files are present on disk; download nothing",
    )

    args = p.parse_args(argv)

    if args.num_shards < 1:
        p.error(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        p.error(
            f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}"
        )
    if args.workers < 1:
        p.error(f"--workers must be >= 1, got {args.workers}")

    if args.data_root is None:
        args.data_root = (
            SNELLIUS_DATA_ROOT if args.env == "snellius" else LOCAL_DATA_ROOT
        )
    args.data_root = args.data_root.expanduser().resolve()

    if args.failure_log is None:
        args.failure_log = args.data_root / "logs" / f"failures_{args.shard_id}.json"

    return args


def setup_hf_env(data_root: Path, no_fast_transfer: bool) -> None:
    """Set HF env vars.

    Must run before ``huggingface_hub`` is imported: its ``constants`` module reads
    these at import time.  ``cache_dir`` is still passed explicitly at every call site
    so behaviour never depends on that ordering.

    Pointing HF_HOME at the data root also moves the Xet chunk cache there
    (``<data-root>/xet``), which matters on Snellius: it can grow to tens of GB and
    home directories are quota'd.

    Transfers go through Xet, not the old ``hf_transfer`` downloader — huggingface_hub
    ignores ``HF_HUB_ENABLE_HF_TRANSFER`` as of v1.
    """
    os.environ["HF_HOME"] = str(data_root)

    if no_fast_transfer:
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
        return

    try:
        import hf_xet  # noqa: F401
    except ImportError:
        print(
            "warning: hf_xet not installed, falling back to plain HTTP "
            "(pip install hf_xet for faster downloads)",
            file=sys.stderr,
        )
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    else:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def select_files(
    api: "HfApi",
    sha: str,
    configs: Sequence[str],
    splits: Sequence[str],
) -> tuple[list[str], dict[str, int], list[tuple[str, str]]]:
    """Return (sorted parquet paths, {path: size}, skipped (config, split) pairs).

    The config -> split -> path-glob map comes from the dataset card rather than being
    hardcoded, so it survives the repo being re-sharded.  Sizes come along for free
    from the same tree listing and drive the job wall-time estimate.
    """
    from huggingface_hub.utils import filter_repo_objects

    card_configs = api.dataset_info(REPO_ID, revision=sha).cardData["configs"]
    globs_by_config = {
        entry["config_name"]: {
            f["split"]: f["path"] for f in entry["data_files"]
        }
        for entry in card_configs
    }

    patterns, skipped = [], []
    for config in configs:
        available = globs_by_config.get(config, {})
        for split in splits:
            if split in available:
                patterns.append(available[split])
            else:
                skipped.append((config, split))

    sizes = {
        entry.path: entry.size
        for entry in api.list_repo_tree(
            REPO_ID, repo_type="dataset", revision=sha, recursive=True
        )
        if entry.path.endswith(".parquet")
    }
    files = sorted(filter_repo_objects(sizes, allow_patterns=patterns))
    return files, sizes, skipped


def missing_files(data_root: Path, sha: str, files: Sequence[str]) -> list[str]:
    """Return the subset of ``files`` not yet present in the local cache."""
    snapshot = (
        data_root
        / "hub"
        / f"datasets--{REPO_ID.replace('/', '--')}"
        / "snapshots"
        / sha
    )
    # Symlinks into blobs/, so resolve before checking rather than trusting the link.
    return [f for f in files if not (snapshot / f).exists()]


def download(sha: str, files: Sequence[str], args: argparse.Namespace) -> list[str]:
    """Download ``files``, retrying on failure.  Returns what is still missing."""
    from huggingface_hub import snapshot_download

    cache_dir = args.data_root / "hub"

    for attempt in range(1, args.max_retries + 1):
        try:
            snapshot_download(
                REPO_ID,
                repo_type="dataset",
                revision=sha,
                cache_dir=cache_dir,
                allow_patterns=files,
                max_workers=args.workers,
            )
        except Exception as exc:
            print(
                f"attempt {attempt}/{args.max_retries} failed: {exc}", file=sys.stderr
            )
            if attempt < args.max_retries:
                backoff = 2**attempt
                print(f"retrying in {backoff}s", file=sys.stderr)
                time.sleep(backoff)
            continue

        # Downloads resume, so a clean return still gets verified against disk.
        missing = missing_files(args.data_root, sha, files)
        if not missing:
            return []
        print(
            f"attempt {attempt}/{args.max_retries}: {len(missing)} file(s) still missing",
            file=sys.stderr,
        )

    return missing_files(args.data_root, sha, files)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the download, returning a process exit code."""
    args = parse_args(argv)
    setup_hf_env(args.data_root, args.no_fast_transfer)

    # Imported here, after setup_hf_env: huggingface_hub freezes env config at import.
    from huggingface_hub import HfApi

    api = HfApi()
    sha = api.dataset_info(REPO_ID, revision=args.revision).sha

    files, sizes, skipped = select_files(api, sha, args.configs, args.splits)
    for config, split in skipped:
        print(f"warning: {config}/{split} does not exist, skipping", file=sys.stderr)
    if not files:
        print("error: no files selected", file=sys.stderr)
        return 1

    shard_files = files[args.shard_id :: args.num_shards]
    mine = METADATA_FILES + shard_files

    print(f"data root   {args.data_root}")
    print(f"revision    {sha}")
    print(f"configs     {' '.join(args.configs)}")
    print(f"splits      {' '.join(args.splits)}")
    print(
        f"shard       {args.shard_id}/{args.num_shards} "
        f"-> {len(shard_files)} of {len(files)} shard(s)"
    )
    print(f"workers     {args.workers}")
    print(f"xet fast    {'on' if os.environ.get('HF_XET_HIGH_PERFORMANCE') else 'off'}")

    # Machine-readable; submit_librispeech.sh reads these to size the sbatch --time.
    # Shard 0 takes ceil(n/num_shards) files, so it is the worst case.
    shard_bytes = sum(sizes[f] for f in shard_files)
    total_bytes = sum(sizes[f] for f in files)
    print(f"shard bytes {shard_bytes}")
    print(f"total bytes {total_bytes}")
    print(f"size        {shard_bytes / 1e9:.2f} GB of {total_bytes / 1e9:.2f} GB")

    if args.dry_run:
        for f in mine:
            print(f)
        return 0

    if args.verify:
        missing = missing_files(args.data_root, sha, mine)
        if missing:
            print(f"MISSING {len(missing)} of {len(mine)} file(s):", file=sys.stderr)
            for f in missing:
                print(f"  {f}", file=sys.stderr)
            return 1
        print(f"OK: all {len(mine)} file(s) present")
        return 0

    args.data_root.mkdir(parents=True, exist_ok=True)
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)

    missing = download(sha, mine, args)

    if missing:
        # Per-task log file: array tasks must never write the same path.
        args.failure_log.write_text(
            json.dumps(
                {
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "revision": sha,
                    "missing": missing,
                },
                indent=2,
            )
        )
        print(
            f"FAILED: {len(missing)} of {len(mine)} file(s) missing after "
            f"{args.max_retries} attempt(s), logged to {args.failure_log}",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(mine)} file(s) downloaded to {args.data_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
