"""Verify that a sharded eval array produced complete, non-overlapping results.

Checks the properties that only break at scale — a task silently writing nothing, two
tasks claiming the same utterances, a split missing entirely — none of which the WER
number itself would reveal. A rehearsal run through this before the real submission turns
"the array looked fine" into an assertion.

    python -m scripts.evaluation.check_eval --results-dir <dir> --expect-total 32 --num-shards 8
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--num-shards", type=int, required=True, help="array task count")
    p.add_argument(
        "--expect-total",
        type=int,
        default=None,
        help="utterances expected per split (the --limit used, or the split size)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="dataset keys the run covered; one manifest is expected per key",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check the sharded output, returning a process exit code."""
    args = parse_args(argv)
    n_expected_manifests = len(args.datasets)

    parts = sorted(args.results_dir.glob("*.part*.jsonl"))
    if not parts:
        print(f"FAIL: no .partN.jsonl files under {args.results_dir}", file=sys.stderr)
        return 1

    # Group parts by the manifest they belong to.
    groups: dict[str, list[Path]] = {}
    for part in parts:
        groups.setdefault(re.sub(r"\.part\d+\.jsonl$", "", part.name), []).append(part)

    failures: list[str] = []
    print(f"{'manifest':52s}{'parts':>7s}{'rows':>7s}{'uniq':>7s}")
    print("-" * 73)

    for name, group in sorted(groups.items()):
        shard_ids = sorted(
            int(re.search(r"\.part(\d+)\.jsonl$", p.name).group(1)) for p in group
        )
        ids: list[str] = []
        empty_parts = []
        for part in sorted(group):
            with part.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            if not rows:
                empty_parts.append(part.name)
            ids.extend(r["audio_filepath"] for r in rows)

        counts = Counter(ids)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        short = name.split("DATASET_")[-1]
        print(f"{short[:51]:52s}{len(group):7d}{len(ids):7d}{len(counts):7d}")

        # Every task must have contributed a part, or one silently did nothing.
        missing_shards = sorted(set(range(args.num_shards)) - set(shard_ids))
        if missing_shards:
            failures.append(f"{short}: no part from task(s) {missing_shards}")
        if empty_parts:
            failures.append(f"{short}: empty part(s) {empty_parts}")
        # Overlap means shards were computed against a different --num-shards.
        if duplicates:
            failures.append(
                f"{short}: {len(duplicates)} duplicate id(s), e.g. "
                f"{sorted(duplicates)[:3]} — shards overlap"
            )
        if args.expect_total is not None and len(counts) != args.expect_total:
            failures.append(
                f"{short}: {len(counts)} unique utterances, expected {args.expect_total}"
            )

    if len(groups) != n_expected_manifests:
        failures.append(
            f"{len(groups)} manifest(s) but {n_expected_manifests} expected "
            f"({' '.join(args.datasets)}) — a dataset produced nothing"
        )

    print("-" * 73)
    if failures:
        print(f"\nFAIL ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"\nPASS: {len(groups)} manifest(s), all {args.num_shards} shards present, "
        "no overlap, counts as expected"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
