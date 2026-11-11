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
stronger evidence the harness is right than matching either alone. Both columns are
restricted to ``--model``'s manifests, since comparing two different runs would make the
agreement meaningless.

``--results-dir``, ``--model`` and ``--num-shards`` must describe the run that produced
the manifests; jobs/batch/score_eval.sh threads them through from the same env vars
submit/eval.sh used.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

from src.constants import REPO_ROOT

LEADERBOARD = REPO_ROOT / "open_asr_leaderboard"

# The checkpoint LIVE_TARGETS and CSV_TARGETS were read for. --model is free, so the
# targets are only asserted when the run used this checkpoint; anything else reports.
TARGETS_MODEL = "openai/whisper-large-v3"

# Live leaderboard figures for openai/whisper-large-v3, read by hand from the Gradio Space
# (it is a client-rendered app and cannot be fetched programmatically). A value of None
# means "not yet read off the Space": the column still prints, it just is not asserted.
LIVE_TARGETS: dict[str, float | None] = {
    "librispeech_test.clean": 1.56,
    "librispeech_test.other": 3.52,
    "gigaspeech_cleaned_test": 8.35,
    "ami_cleaned_test": 13.63,
    "spgispeech_test": 2.71,
    "voxpopuli_cleaned_aa_test": 4.53,
    "earnings22_cleaned_aa_chunked_test": 7.99,
}
# Derived from the per-dataset targets rather than hardcoded, so the two can never drift
# apart. Equals 6.04 for the seven columns above.
AVERAGE_TARGET = sum(v for v in LIVE_TARGETS.values() if v is not None) / max(
    sum(v is not None for v in LIVE_TARGETS.values()), 1
)

# Superseded CSV (scripts/data/en_shortform.csv, 2026-05-22), pre-kaldialign — useful only
# where the dataset is defined identically in both eras. Deliberately excludes the *_cleaned
# variants: the old CSV's "Gigaspeech 10.02" is plain gigaspeech, a different set from
# gigaspeech_cleaned, so pairing them would invite a meaningless comparison.
CSV_TARGETS = {"librispeech_test.clean": 2.01, "librispeech_test.other": 3.91}


def add_leaderboard_to_path() -> None:
    """Put the pinned submodule on sys.path so `normalizer` is importable."""
    if not (LEADERBOARD / "normalizer" / "__init__.py").is_file():
        raise SystemExit(
            f"{LEADERBOARD} is empty — run: git submodule update --init --recursive"
        )
    if str(LEADERBOARD) not in sys.path:
        sys.path.insert(0, str(LEADERBOARD))


def part_index(path: Path) -> int:
    """Shard index encoded in a ``*.partN.jsonl`` filename."""
    return int(re.search(r"\.part(\d+)\.jsonl$", path.name).group(1))


def merge_parts(
    results_dir: Path, num_shards: int | None = None, model: str | None = None
) -> int:
    """Concatenate ``*.partN.jsonl`` shards into their canonical manifest.

    A sharded array run writes one file per task. score_results derives the dataset name
    from the filename, so left alone it would score each part as its own dataset and
    average them — which is not the same number as scoring the union. Returns how many
    canonical manifests were produced.

    Merging is destructive (the parts are unlinked), so an incomplete set is refused
    before anything is written: scoring the union of a partially-failed array produces a
    plausible-looking WER over fewer utterances and destroys the evidence that it was
    partial. Without ``num_shards`` only gaps below the highest index are detectable —
    a missing final task looks exactly like a smaller array.

    ``model`` restricts the merge to that checkpoint's parts. Each checkpoint's parts
    group under their own canonical name, so merging all of them mixes no numbers, but it
    would consume and delete the parts of a run that is not being scored.
    """
    groups: dict[Path, list[Path]] = {}
    for part in sorted(results_dir.glob("*.part*.jsonl")):
        if model is not None and not belongs_to_model(str(part), model):
            continue
        canonical = results_dir / (re.sub(r"\.part\d+\.jsonl$", "", part.name) + ".jsonl")
        groups.setdefault(canonical, []).append(part)

    for canonical, parts in groups.items():
        present = {part_index(p) for p in parts}
        expected = num_shards if num_shards is not None else max(present) + 1
        missing = sorted(set(range(expected)) - present)
        if missing:
            raise SystemExit(
                f"{canonical.name}: missing part(s) {missing} of {expected}. Those array "
                "tasks did not finish — check their slurm_eval_*.out. Merging now would "
                "score a partial run and delete the parts; rerun the missing shards, or "
                "pass --num-shards to state the array size explicitly."
            )

        # Sorted by shard index, not lexically, so part10 does not land before part2.
        parts.sort(key=part_index)
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


def belongs_to_model(path: str, model: str) -> bool:
    """Whether a manifest is this model's, by the same test score_results applies.

    Mirrored rather than imported because it is inlined in score_results. The two columns
    must agree on the file set or the cross-check compares different runs.
    """
    slug = model.replace("/", "-")
    return f"/{slug}/" in path or f"MODEL_{slug}_DATASET_" in path


def jiwer_wer_by_dataset(results_dir: Path, model: str) -> dict[str, float]:
    """Plain jiwer WER per manifest, over the same normalized text score_results uses."""
    import jiwer
    from normalizer import data_utils, eval_utils

    out = {}
    for path in sorted(glob.glob(f"{results_dir}/**/*.jsonl", recursive=True)):
        if not belongs_to_model(path, model):
            continue
        with open(path, encoding="utf-8") as fh:
            manifest = [json.loads(line) for line in fh if line.strip()]
        if not manifest:
            continue
        # Same collapse score_results performs: a chunked dataset stores one row per chunk
        # but one reference per parent call, so scoring the rows directly compares each
        # chunk against the whole call — which reads as ~98% WER rather than ~8%.
        manifest = eval_utils.merge_chunked_manifest(manifest)
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
    p.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="array task count the run used; lets the merge detect a missing final shard",
    )
    args = p.parse_args(argv)
    # score_results reads an empty model_id as "do not filter", which would pool every
    # checkpoint in the directory into one WER instead of failing.
    if not args.model:
        p.error("--model must not be empty")
    if args.num_shards is not None and args.num_shards < 1:
        p.error(f"--num-shards must be >= 1, got {args.num_shards}")
    args.results_dir = args.results_dir.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    """Score the manifests and check them against the targets."""
    args = parse_args(argv)
    add_leaderboard_to_path()

    if not glob.glob(f"{args.results_dir}/**/*.jsonl", recursive=True):
        print(f"error: no manifests under {args.results_dir}", file=sys.stderr)
        return 2

    merge_parts(args.results_dir, args.num_shards, args.model)

    from normalizer import eval_utils

    _composite, results = eval_utils.score_results(str(args.results_dir), args.model)
    jiwer_wers = jiwer_wer_by_dataset(args.results_dir, args.model)

    # results keys are "model | dataset_id"; reduce to dataset_id for comparison.
    kaldi = {key.split(" | ", 1)[1]: val["wer"] for key, val in results.items()}

    # The built-in targets are one checkpoint's numbers. Asserting them against a
    # different model would fail for the only reason that is not a bug, so a non-default
    # model reports unless --expect states what to compare against.
    if args.expect:
        expected = parse_expect(args.expect)
    elif args.model == TARGETS_MODEL:
        expected = LIVE_TARGETS
    else:
        print(
            f"note: --model {args.model} is not {TARGETS_MODEL}, which the built-in "
            "targets were read for — reporting without assertions. Pass --expect "
            "dataset:split=wer to assert against this model's own numbers.",
            file=sys.stderr,
        )
        expected = {}

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
        # The old CSV is the same checkpoint's history; against another model it is not a
        # comparison, just a number in the same row.
        old = (
            next((v for k, v in CSV_TARGETS.items() if k in dataset_id), None)
            if args.model == TARGETS_MODEL
            else None
        )
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
    print("-" * 78)

    # Average across the evaluated columns. Only comparable to the target once every
    # column is present, so a partial run reports the shortfall rather than printing a
    # smaller mean that looks like the real thing.
    scored = [kaldi[d] for d in sorted(kaldi)]
    complete = len(scored) == len(LIVE_TARGETS)
    # AVERAGE_TARGET is derived from LIVE_TARGETS, so it means nothing once the targets
    # are another model's or --expect replaced them.
    comparable = expected is LIVE_TARGETS
    if scored:
        avg = sum(scored) / len(scored)
        note = "" if complete else f"   partial: {len(scored)}/{len(LIVE_TARGETS)} sets"
        shown = f"{AVERAGE_TARGET:9.2f}" if comparable else f"{'-':>9s}"
        row = f"{'AVERAGE':38s}{avg:11.2f}%{shown}{'':18s}{note}"
        if comparable and complete and abs(avg - AVERAGE_TARGET) > args.tolerance:
            row += f"   OFF by {abs(avg - AVERAGE_TARGET):.2f}"
            failures.append(("AVERAGE", avg, AVERAGE_TARGET))
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
