"""Validate every eval dataset loads and reads correctly, cheaply and locally.

Fetches only the **smallest shard** of each config and reads a handful of rows from it, so
a schema surprise costs ~1 GB and two minutes here instead of a queued cluster job.

Deliberately avoids ``load_dataset(repo, config, split=...)``: that resolves the *config*
before the split and pulls every shard in it — 11.4 GB for spgispeech. Shards are fetched
with ``snapshot_download(allow_patterns=[...])`` and read via the parquet loader directly,
the same pattern ``scripts.download_librispeech`` uses for the same reason.

    python -m scripts.smoke_datasets
    python -m scripts.smoke_datasets --datasets ami_cleaned --rows 8
"""

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi

from scripts.download_librispeech import LOCAL_DATA_ROOT, SNELLIUS_DATA_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = REPO_ROOT / "open_asr_leaderboard"

EVAL_REPO = "hf-audio/open-asr-leaderboard"
# The chunked set lives in its own repos and needs its parent transcripts to be readable.
CHUNKED_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
CHUNKED_PARENT_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA"

DATASETS = [
    "ami_cleaned",
    "gigaspeech_cleaned",
    "spgispeech",
    "voxpopuli_cleaned_aa",
    "librispeech",
]
EXPECTED_SAMPLING_RATE = 16_000


def add_leaderboard_to_path() -> None:
    """Put the pinned submodule on sys.path so `normalizer` is importable."""
    if not (LEADERBOARD / "normalizer" / "__init__.py").is_file():
        raise SystemExit(
            f"{LEADERBOARD} is empty — run: git submodule update --init --recursive"
        )
    if str(LEADERBOARD) not in sys.path:
        sys.path.insert(0, str(LEADERBOARD))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--data-root", type=Path, default=None, help="overrides --env")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--rows", type=int, default=4, help="rows to read per dataset")
    p.add_argument(
        "--skip-chunked",
        action="store_true",
        help="skip earnings22, whose audio is fetched whole rather than per shard",
    )
    args = p.parse_args(argv)
    if args.data_root is None:
        args.data_root = SNELLIUS_DATA_ROOT if args.env == "snellius" else LOCAL_DATA_ROOT
    args.data_root = args.data_root.expanduser().resolve()
    return args


def smallest_shard(api: "HfApi", config: str) -> tuple[str, int]:
    """Return the (path, size) of the smallest parquet shard for one config."""
    files = [
        f
        for f in api.list_repo_tree(EVAL_REPO, config, repo_type="dataset", recursive=True)
        if getattr(f, "size", None) and f.path.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(f"no parquet shards under {config}/ in {EVAL_REPO}")
    smallest = min(files, key=lambda f: f.size)
    return smallest.path, smallest.size


def check_rows(
    rows: list[dict],
    normalizer: Callable[[str], str],
    get_text: Callable[[dict], str],
) -> list[str]:
    """Return a list of problems found in the sample rows; empty means healthy."""
    problems = []
    row = rows[0]

    audio = row.get("audio")
    if audio is None:
        problems.append("no 'audio' column")
    else:
        # datasets v5 decodes this to a torchcodec AudioDecoder, which is subscriptable
        # but is neither a dict nor a Mapping and has no .get — so duck-type it, exactly
        # as scripts.precompute_whisper does.
        try:
            rate = audio["sampling_rate"]
            array = audio["array"]
        except (TypeError, KeyError) as exc:
            problems.append(f"audio not subscriptable ({type(audio).__name__}): {exc}")
        else:
            if rate != EXPECTED_SAMPLING_RATE:
                problems.append(f"sampling_rate {rate}, expected {EXPECTED_SAMPLING_RATE}")
            if array is None or len(array) == 0:
                problems.append("audio decoded to nothing")

    try:
        texts = [get_text(r) for r in rows]
    except ValueError as exc:
        problems.append(f"no transcript column: {exc}")
        return problems

    if not any(t.strip() for t in texts):
        problems.append("all sampled transcripts are empty")
    # An all-empty normalization would silently zero the reference side of every WER.
    if not any(normalizer(t).strip() for t in texts):
        problems.append("normalizer emptied every sampled transcript")
    return problems


def smoke_config(api: "HfApi", config: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Fetch the smallest shard of one config and read rows from it."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from normalizer import data_utils

    path, size = smallest_shard(api, config)
    root = snapshot_download(
        EVAL_REPO,
        repo_type="dataset",
        cache_dir=args.data_root / "hub",
        allow_patterns=[path],
    )
    ds = load_dataset("parquet", data_files=str(Path(root) / path), split="train")
    rows = [ds[i] for i in range(min(args.rows, ds.num_rows))]
    problems = check_rows(rows, data_utils.normalizer, data_utils.get_text)

    detail = f"{size / 1e9:.2f} GB shard, {ds.num_rows} rows, cols={sorted(rows[0])[:4]}"
    if problems:
        return False, detail + " | " + "; ".join(problems)
    return True, detail


def smoke_chunked(args: argparse.Namespace) -> tuple[bool, str]:
    """Check the chunked earnings22 set: metadata, parent transcripts, and audio."""
    from datasets import load_dataset
    from normalizer import data_utils

    parent = load_dataset(CHUNKED_PARENT_REPO, split="test")
    chunks = load_dataset(CHUNKED_REPO, split="test")

    parent_ids = {r["id"] for r in parent}
    missing = sorted({r["parent_id"] for r in chunks} - parent_ids)
    cols = sorted(chunks[0])
    problems = []
    if missing:
        problems.append(f"{len(missing)} chunk parent_id(s) absent from the parent repo")
    for key in data_utils.CHUNK_METADATA_KEYS:
        if key not in cols:
            problems.append(f"missing chunk metadata column {key!r}")

    detail = f"{chunks.num_rows} chunks over {parent.num_rows} calls, cols={cols[:4]}"
    if problems:
        return False, detail + " | " + "; ".join(problems)
    return True, detail


def main(argv: list[str] | None = None) -> int:
    """Smoke-test every dataset, returning a process exit code."""
    args = parse_args(argv)
    os.environ["HF_HOME"] = str(args.data_root)
    add_leaderboard_to_path()

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"data root {args.data_root}")
    print(f"reading {args.rows} row(s) from the smallest shard of each config\n")
    print(f"{'dataset':24s}{'result':8s}detail")
    print("-" * 100)

    failures = 0
    for config in args.datasets:
        try:
            ok, detail = smoke_config(api, config, args)
        except Exception as exc:  # a 403 on a gated set is a normal outcome here
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:70]}"
        failures += not ok
        print(f"{config:24s}{'PASS' if ok else 'FAIL':8s}{detail}")

    if not args.skip_chunked:
        try:
            ok, detail = smoke_chunked(args)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:70]}"
        failures += not ok
        print(f"{'earnings22_chunked':24s}{'PASS' if ok else 'FAIL':8s}{detail}")

    print("-" * 100)
    if failures:
        print(
            f"\nFAIL: {failures} dataset(s) unusable — fix before submitting a job",
            file=sys.stderr,
        )
        return 1
    print("\nPASS: every dataset loads, decodes at 16 kHz, and normalizes to non-empty text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
