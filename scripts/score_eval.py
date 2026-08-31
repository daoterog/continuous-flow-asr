"""Score eval manifests with the Open ASR Leaderboard's normalizer and WER.

The leaderboard's ``score_results`` is called unchanged — normalization is the thing being
validated, so reimplementing it would defeat the purpose.

Two WER numbers are reported per split, because the repo contains two implementations and
they are not interchangeable:

* **kaldialign** — ``batch_error_rate(..., merge_compounds=True)``, what ``score_results``
  uses today, and what the live leaderboard figures were produced with. This is the column
  ``--expect`` asserts on.
* **jiwer** — plain WER over the same normalized text. It matches the older published CSV
  (LS Clean 2.01), which predates the kaldialign commit by three weeks.

They agree on ordinary substitutions and deletions and diverge only on compound-word
boundaries, which ``merge_compounds`` forgives. Seeing both eras line up at once is much
stronger evidence the harness is right than matching either alone.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = REPO_ROOT / "open_asr_leaderboard"

# Live leaderboard figures for openai/whisper-large-v3, read from the Gradio Space.
LIVE_TARGETS = {"librispeech_test.clean": 1.56, "librispeech_test.other": 3.52}
# Superseded CSV (scripts/data/en_shortform.csv, 2026-05-22), pre-kaldialign.
CSV_TARGETS = {"librispeech_test.clean": 2.01, "librispeech_test.other": 3.91}


def add_leaderboard_to_path() -> None:
    """Put the pinned submodule on sys.path so `normalizer` is importable."""
    if not (LEADERBOARD / "normalizer" / "__init__.py").is_file():
        raise SystemExit(
            f"{LEADERBOARD} is empty — run: git submodule update --init --recursive"
        )
    if str(LEADERBOARD) not in sys.path:
        sys.path.insert(0, str(LEADERBOARD))


def merge_parts(results_dir: Path) -> int:
    """Concatenate ``*.partN.jsonl`` shards into their canonical manifest.

    A sharded array run writes one file per task. score_results derives the dataset name
    from the filename, so left alone it would score each part as its own dataset and
    average them — which is not the same number as scoring the union. Returns how many
    canonical manifests were produced.
    """
    groups: dict[Path, list[Path]] = {}
    for part in sorted(results_dir.glob("*.part*.jsonl")):
        canonical = results_dir / (re.sub(r"\.part\d+\.jsonl$", "", part.name) + ".jsonl")
        groups.setdefault(canonical, []).append(part)

    for canonical, parts in groups.items():
        # Sorted by shard index, not lexically, so part10 does not land before part2.
        parts.sort(key=lambda p: int(re.search(r"\.part(\d+)\.jsonl$", p.name).group(1)))
        rows = 0
        seen: set[str] = set()
        duplicates: set[str] = set()
        with canonical.open("w", encoding="utf-8") as out:
            for part in parts:
                with part.open(encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        utt_id = json.loads(line)["audio_filepath"]
                        if utt_id in seen:
                            duplicates.add(utt_id)
                        seen.add(utt_id)
                        out.write(line)
                        rows += 1
        print(f"merged {len(parts)} part(s), {rows} rows -> {canonical.name}")
        if duplicates:
            # Overlapping shards would double-count utterances and skew WER, so this is
            # an error rather than a warning.
            raise SystemExit(
                f"{canonical.name}: {len(duplicates)} duplicate utterance id(s) across "
                f"parts, e.g. {sorted(duplicates)[:3]}. The shards overlap — check "
                "--num-shards matched the array size on every task."
            )
        for part in parts:
            part.unlink()
    return len(groups)


def jiwer_wer_by_dataset(results_dir: Path) -> dict[str, float]:
    """Plain jiwer WER per manifest, over the same normalized text score_results uses."""
    import jiwer
    from normalizer import data_utils

    out = {}
    for path in sorted(glob.glob(f"{results_dir}/**/*.jsonl", recursive=True)):
        with open(path, encoding="utf-8") as fh:
            manifest = [json.loads(line) for line in fh if line.strip()]
        if not manifest:
            continue
        refs = [data_utils.normalizer(d["text"]) for d in manifest]
        preds = [data_utils.normalizer(d["pred_text"]) for d in manifest]
        # Empty references make jiwer raise; the leaderboard filters them at write time.
        pairs = [(r, p) for r, p in zip(refs, preds, strict=True) if r.strip()]
        if not pairs:
            continue
        dataset_id = Path(path).name.split("DATASET_")[-1].removesuffix(".jsonl")
        out[dataset_id] = 100 * jiwer.wer([r for r, _ in pairs], [p for _, p in pairs])
    return out


def parse_expect(items: list[str]) -> dict[str, float]:
    """Parse ``dataset:split=wer`` pairs into a {dataset_key: expected} mapping."""
    expected = {}
    for item in items:
        m = re.fullmatch(r"([\w.\-]+):([\w.\-]+)=([\d.]+)", item)
        if not m:
            raise SystemExit(f"bad --expect entry {item!r}, want dataset:split=wer")
        expected[f"{m.group(1)}_{m.group(2)}"] = float(m.group(3))
    return expected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument(
        "--expect",
        nargs="*",
        default=None,
        help="dataset:split=wer to assert on the kaldialign column; "
        "omit to use the built-in live-leaderboard targets",
    )
    p.add_argument("--tolerance", type=float, default=0.1, help="absolute WER tolerance")
    p.add_argument("--no-assert", action="store_true", help="report only, never fail")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Score the manifests and check them against the targets."""
    args = parse_args(argv)
    add_leaderboard_to_path()

    if not glob.glob(f"{args.results_dir}/**/*.jsonl", recursive=True):
        print(f"error: no manifests under {args.results_dir}", file=sys.stderr)
        return 2

    merge_parts(args.results_dir)

    from normalizer import eval_utils

    _composite, results = eval_utils.score_results(str(args.results_dir), args.model)
    jiwer_wers = jiwer_wer_by_dataset(args.results_dir)

    # results keys are "model | dataset_id"; reduce to dataset_id for comparison.
    kaldi = {key.split(" | ", 1)[1]: val["wer"] for key, val in results.items()}
    expected = parse_expect(args.expect) if args.expect else LIVE_TARGETS

    print("\n" + "=" * 78)
    print(f"{'dataset':38s}{'kaldialign':>12s}{'target':>9s}{'jiwer':>9s}{'old CSV':>9s}")
    print("-" * 78)
    failures = []
    for dataset_id in sorted(kaldi):
        # dataset_id is "<repo>_<config>_<split>"; show the leaderboard key when present
        # ("librispeech_test.clean") rather than the whole mangled repo prefix.
        known = [k for k in (*LIVE_TARGETS, *CSV_TARGETS) if k in dataset_id]
        short = known[0] if known else dataset_id
        target = next((v for k, v in expected.items() if k in dataset_id), None)
        old = next((v for k, v in CSV_TARGETS.items() if k in dataset_id), None)
        row = (
            f"{short[:37]:38s}{kaldi[dataset_id]:11.2f}%"
            f"{('-' if target is None else f'{target:.2f}'):>9s}"
            f"{jiwer_wers.get(dataset_id, float('nan')):8.2f}%"
            f"{('-' if old is None else f'{old:.2f}'):>9s}"
        )
        if target is not None and abs(kaldi[dataset_id] - target) > args.tolerance:
            row += f"   OFF by {abs(kaldi[dataset_id] - target):.2f}"
            failures.append((dataset_id, kaldi[dataset_id], target))
        print(row)
    print("=" * 78)

    if failures:
        print(
            f"\n{len(failures)} split(s) outside +/-{args.tolerance} of target.\n"
            "Check, in order: is the target the whisper-large-v3 row? are generation\n"
            "kwargs (language/task, greedy) right? is the dataset revision the same?\n"
            "The normalizer is used verbatim, so it is the least likely cause.",
            file=sys.stderr,
        )
        if not args.no_assert:
            return 1
        print("(--no-assert: reporting only)")
        return 0

    print("\nOK: all scored splits within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
