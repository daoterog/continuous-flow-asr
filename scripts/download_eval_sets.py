"""Download the Open ASR Leaderboard English short-form eval sets.

Fetches whole configs with ``snapshot_download`` and shard-level ``allow_patterns``, never
``load_dataset(repo, config, split=...)`` — that resolves the config before the split and
pulls every shard in it. The eval itself *does* call ``data_utils.load_data``, which is a
pure cache hit once a config is fully downloaded; pre-fetching here is what makes that safe,
and stops eight array tasks racing on the same cold cache.

Work parallelises across SLURM array tasks at the file level: task ``k`` of ``N`` takes
``files[k::N]``, the same scheme as ``scripts.download_librispeech``.

    python -m scripts.download_eval_sets --env snellius
    python -m scripts.download_eval_sets --datasets voxpopuli_cleaned_aa --verify
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.download_librispeech import LOCAL_DATA_ROOT, SNELLIUS_DATA_ROOT

if TYPE_CHECKING:
    from huggingface_hub import HfApi

EVAL_REPO = "hf-audio/open-asr-leaderboard"
# The chunked set lives in its own repos: the chunks, and the parent call transcripts that
# data_utils.load_chunked_data joins onto them.
CHUNKED_REPOS = [
    "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked",
    "ArtificialAnalysis/Earnings22-Cleaned-AA",
]
CHUNKED_KEY = "earnings22_cleaned_aa_chunked"

# librispeech is omitted: already downloaded from openslr, and its normalized references
# were verified identical (2620/2620) to this repo's copy.
DATASETS = [
    "ami_cleaned",
    "gigaspeech_cleaned",
    "spgispeech",
    "voxpopuli_cleaned_aa",
    CHUNKED_KEY,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--data-root", type=Path, default=None, help="overrides --env")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
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
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--failure-log", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--verify", action="store_true", help="check files exist; download nothing")

    args = p.parse_args(argv)
    if args.num_shards < 1:
        p.error(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        p.error(f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}")
    if args.data_root is None:
        args.data_root = SNELLIUS_DATA_ROOT if args.env == "snellius" else LOCAL_DATA_ROOT
    args.data_root = args.data_root.expanduser().resolve()
    if args.failure_log is None:
        args.failure_log = args.data_root / "logs" / f"eval_dl_failures_{args.shard_id}.json"
    return args


def select_files(api: "HfApi", datasets: Sequence[str]) -> dict[str, list[tuple[str, int]]]:
    """Return {repo: [(path, size)]} for every file the selected datasets need."""
    plan: dict[str, list[tuple[str, int]]] = {}

    configs = [d for d in datasets if d != CHUNKED_KEY]
    if configs:
        entries = [
            (f.path, f.size)
            for f in api.list_repo_tree(EVAL_REPO, repo_type="dataset", recursive=True)
            if getattr(f, "size", None) and f.path.split("/")[0] in configs
        ]
        # The dataset card carries the config definitions; without it load_dataset cannot
        # resolve a config name at all.
        entries.append(("README.md", 0))
        plan[EVAL_REPO] = sorted(entries)

    if CHUNKED_KEY in datasets:
        for repo in CHUNKED_REPOS:
            plan[repo] = sorted(
                (f.path, f.size)
                for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
                if getattr(f, "size", None)
            )
    return plan


def missing_files(cache_dir: Path, repo: str, files: Sequence[str]) -> list[str]:
    """Return the subset of ``files`` not present in the local cache for ``repo``."""
    base = cache_dir / f"datasets--{repo.replace('/', '--')}" / "snapshots"
    snapshots = sorted(p for p in base.glob("*") if p.is_dir()) if base.is_dir() else []
    if not snapshots:
        return list(files)
    # Symlinks into blobs/, so .exists() resolves them rather than trusting the link.
    return [f for f in files if not any((s / f).exists() for s in snapshots)]


def download(repo: str, files: Sequence[str], args: argparse.Namespace) -> list[str]:
    """Download ``files`` from one repo, retrying. Returns what is still missing."""
    from huggingface_hub import snapshot_download

    cache_dir = args.data_root / "hub"
    for attempt in range(1, args.max_retries + 1):
        try:
            snapshot_download(
                repo,
                repo_type="dataset",
                cache_dir=cache_dir,
                allow_patterns=list(files),
                max_workers=args.workers,
            )
        except Exception as exc:  # any failure is worth a retry; downloads resume
            print(f"  attempt {attempt}/{args.max_retries} failed: {exc}", file=sys.stderr)
            if attempt < args.max_retries:
                time.sleep(2**attempt)
            continue
        missing = missing_files(cache_dir, repo, files)
        if not missing:
            return []
        print(f"  attempt {attempt}: {len(missing)} file(s) still missing", file=sys.stderr)
    return missing_files(cache_dir, repo, files)


def main(argv: list[str] | None = None) -> int:
    """Download the selected eval sets, returning a process exit code."""
    args = parse_args(argv)
    # Before any hub import: these are tens of GB and must not land in a quota'd home.
    os.environ["HF_HOME"] = str(args.data_root)

    from huggingface_hub import HfApi

    plan = select_files(HfApi(), args.datasets)
    cache_dir = args.data_root / "hub"

    print(f"data root   {args.data_root}")
    print(f"datasets    {' '.join(args.datasets)}")

    total_bytes = 0
    mine: dict[str, list[str]] = {}
    for repo, entries in plan.items():
        slice_ = entries[args.shard_id :: args.num_shards]
        mine[repo] = [path for path, _ in slice_]
        total_bytes += sum(size for _, size in slice_)
        print(f"  {repo:52s} {len(slice_):4d} of {len(entries):4d} file(s)")

    print(f"shard       {args.shard_id}/{args.num_shards}")
    print(f"bytes       {total_bytes}")
    print(f"size        {total_bytes / 1e9:.2f} GB")

    if args.dry_run:
        for repo, files in mine.items():
            for f in files:
                print(f"{repo}\t{f}")
        return 0

    if args.verify:
        incomplete = {r: missing_files(cache_dir, r, f) for r, f in mine.items()}
        incomplete = {r: f for r, f in incomplete.items() if f}
        if incomplete:
            for repo, files in incomplete.items():
                print(f"MISSING {len(files)} file(s) from {repo}:", file=sys.stderr)
                for f in files[:5]:
                    print(f"  {f}", file=sys.stderr)
            return 1
        print(f"OK: all {sum(len(f) for f in mine.values())} file(s) present")
        return 0

    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    failures = {}
    for repo, files in mine.items():
        if not files:
            continue
        print(f"\n[{repo}] {len(files)} file(s)", flush=True)
        still_missing = download(repo, files, args)
        if still_missing:
            failures[repo] = still_missing

    if failures:
        args.failure_log.write_text(
            json.dumps(
                {"shard_id": args.shard_id, "num_shards": args.num_shards, "missing": failures},
                indent=2,
            )
        )
        n = sum(len(f) for f in failures.values())
        print(f"\nFAILED: {n} file(s) missing, logged to {args.failure_log}", file=sys.stderr)
        return 1

    print(f"\nOK: downloaded to {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
