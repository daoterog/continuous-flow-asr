"""Decode cached Whisper encoder latents into transcripts and write result manifests.

This deliberately does **not** run the encoder. It reads the ``.npy``/``.json`` feature
cache written by ``scripts.precompute_whisper`` and feeds those latents straight to the
decoder via ``generate(encoder_outputs=...)``, so a matching WER validates the cache, the
decode path and the normalization together rather than just re-running Whisper.

Decoding from cached latents is exact only when the latents are **untrimmed**. Whisper
pads every clip to a 30 s window and its decoder cross-attends over all 1500 encoder
frames; hand it the ~175 real frames of a short utterance and it emits ``<|en|>``
repeatedly instead of text. Zero-padding trimmed latents back to 1500 recovers the right
answer most of the time but not always (measured: 44/48 transcripts identical), so this
script refuses trimmed caches rather than silently scoring an approximation.

Predictions and references are written **raw**; the leaderboard's scorer normalizes at
scoring time. Score with ``python -m scripts.score_eval``.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.download_librispeech import LOCAL_DATA_ROOT, SNELLIUS_DATA_ROOT

if TYPE_CHECKING:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = REPO_ROOT / "open_asr_leaderboard"

DEFAULT_MODEL = "openai/whisper-large-v3"
# The cache is keyed by openslr's (config, split); the leaderboard scorer keys off
# "librispeech_test.clean" / "librispeech_test.other" substrings in the manifest
# filename. This maps one onto the other so score_results populates its CSV columns.
LEADERBOARD_NAMES = {
    ("clean", "test"): ("librispeech", "test.clean"),
    ("other", "test"): ("librispeech", "test.other"),
    ("clean", "validation"): ("librispeech", "validation.clean"),
    ("other", "validation"): ("librispeech", "validation.other"),
}
SOURCE_DATASET_PATH = "openslr/librispeech_asr"


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
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--data-root", type=Path, default=None, help="overrides --env")
    p.add_argument("--configs", nargs="+", default=["clean", "other"])
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--batch-size", type=int, default=16)
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
    p.add_argument("--limit", type=int, default=None, help="stop after N utts per split")
    p.add_argument("--overwrite", action="store_true", help="redo splits already written")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args(argv)

    if args.batch_size < 1:
        p.error(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.num_shards < 1:
        p.error(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        p.error(f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}")
    if args.data_root is None:
        args.data_root = SNELLIUS_DATA_ROOT if args.env == "snellius" else LOCAL_DATA_ROOT
    args.data_root = args.data_root.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    return args


def feature_dir(data_root: Path, model: str, config: str, split: str) -> Path:
    """Location of the cached latents for one (config, split)."""
    return data_root / "features" / model.split("/")[-1] / config / split


def manifest_name(model: str, dataset: str, split: str, part: int | None = None) -> str:
    """Reproduce the leaderboard's manifest filename, which its scorer parses.

    A sharded run writes ``.partN`` files instead; score_eval concatenates them back
    into this canonical name before scoring, since score_results would otherwise treat
    each part as a separate dataset and average them wrongly.
    """
    stem = (
        f"MODEL_{model.replace('/', '-')}"
        f"_DATASET_{SOURCE_DATASET_PATH.replace('/', '-')}_{dataset}_{split}"
    )
    return f"{stem}.part{part}.jsonl" if part is not None else f"{stem}.jsonl"


def load_decoder(
    model_id: str,
) -> tuple["WhisperForConditionalGeneration", "WhisperProcessor", "torch.device", dict]:
    """Load the model and processor, with generation settings matching the leaderboard."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(model_id)
    model = (
        WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
        .to(device)
        .eval()
    )

    gen_kwargs: dict = {}
    if getattr(model.generation_config, "is_multilingual", False):
        gen_kwargs["language"] = "en"
        gen_kwargs["task"] = "transcribe"

    # Cleared to match the leaderboard's setup exactly. Verified output-neutral on a
    # sample here, but suppress_tokens controls which tokens Whisper may never emit, so
    # matching them removes a source of divergence over the full set.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    return model, processor, device, gen_kwargs


def decode_split(
    args: argparse.Namespace,
    model: "WhisperForConditionalGeneration",
    processor: "WhisperProcessor",
    device: "torch.device",
    gen_kwargs: dict,
    config: str,
    split: str,
) -> str:
    """Decode one cached split and write its manifest. Returns the manifest path."""
    import numpy as np
    import torch
    from normalizer import data_utils
    from transformers.modeling_outputs import BaseModelOutput

    src = feature_dir(args.data_root, args.model, config, split)
    indexes = sorted(src.glob("*.json"))
    if not indexes:
        raise FileNotFoundError(
            f"no cached features under {src} — run scripts.precompute_whisper "
            f"--configs {config} --splits {split} --no-trim"
        )

    refs: list[str] = []
    preds: list[str] = []
    for index_path in indexes:
        meta = json.loads(index_path.read_text())
        if meta["trimmed"]:
            raise ValueError(
                f"{index_path} holds trimmed latents, which Whisper's decoder cannot "
                "decode (it needs all 1500 frames). Re-run precompute with --no-trim."
            )
        feats = np.load(index_path.with_suffix(".npy"), mmap_mode="r")
        utts = meta["utts"][: args.limit] if args.limit else meta["utts"]
        # Interleaved, so every task gets a similar mix of utterance lengths and they
        # finish together; decode time scales with the number of tokens emitted.
        utts = utts[args.shard_id :: args.num_shards]

        for start in range(0, len(utts), args.batch_size):
            chunk = utts[start : start + args.batch_size]
            batch = np.stack(
                [feats[u["offset"] : u["offset"] + u["length"]] for u in chunk]
            )
            hidden = torch.from_numpy(batch).to(device, dtype=model.dtype)
            # Every Whisper encoder output is a full, unpadded 1500 frames, so the mask
            # is all ones. Passing it explicitly only stops generate from warning that it
            # cannot infer one (it has no input_features to derive it from here).
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=device)
            with torch.no_grad():
                ids = model.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=mask,
                    **gen_kwargs,
                )
            # clean_up_tokenization_spaces=False is what transformers already does for a
            # BPE tokenizer (it ignores the checkpoint's True and says so); stating it
            # silences that notice without changing the output.
            preds.extend(
                processor.batch_decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            )
            refs.extend(u["text"] for u in chunk)
            print(f"    {len(preds)}/{len(utts)}", end="\r", flush=True)

    dataset, lb_split = LEADERBOARD_NAMES.get((config, split), ("librispeech", split))

    # audio_length / transcription_time are deliberately omitted: the encoder ran in a
    # separate offline job, so an RTFx measured here would describe the decoder alone and
    # be incomparable to the leaderboard's. score_results skips RTFx when they are absent.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    part = args.shard_id if args.num_shards > 1 else None
    path = args.results_dir / manifest_name(args.model, dataset, lb_split, part)
    with path.open("w", encoding="utf-8") as f:
        for i, (ref, pred) in enumerate(zip(refs, preds, strict=True)):
            f.write(
                json.dumps(
                    {
                        "audio_filepath": f"sample_{i}",
                        "duration": None,
                        "time": None,
                        "text": ref,
                        "pred_text": pred,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    del data_utils  # imported only to fail fast if the submodule is missing
    return str(path)


def main(argv: list[str] | None = None) -> int:
    """Decode the selected cached splits, returning a process exit code."""
    args = parse_args(argv)
    add_leaderboard_to_path()

    pairs = [(c, s) for c in args.configs for s in args.splits]
    pairs = [(c, s) for c, s in pairs if feature_dir(args.data_root, args.model, c, s).is_dir()]
    if not pairs:
        print("error: no cached feature splits matched the selection", file=sys.stderr)
        return 1

    def out_path(config: str, split: str) -> Path:
        dataset, lb_split = LEADERBOARD_NAMES.get((config, split), ("librispeech", split))
        part = args.shard_id if args.num_shards > 1 else None
        return args.results_dir / manifest_name(args.model, dataset, lb_split, part)

    pending = [(c, s) for c, s in pairs if args.overwrite or not out_path(c, s).is_file()]

    print(f"model        {args.model}")
    print(f"features     {args.data_root / 'features' / args.model.split('/')[-1]}")
    print(f"splits       {len(pairs)} -> {len(pending)} pending")
    print(f"results dir  {args.results_dir}")
    print(f"batch size   {args.batch_size}")
    if args.limit:
        print(f"limit        {args.limit} utt(s) per split  (NOT a full run)")

    if args.dry_run:
        for c, s in pairs:
            print(f"  {c}/{s} -> {out_path(c, s).name}")
        return 0
    if not pending:
        print("nothing to do")
        return 0

    model, processor, device, gen_kwargs = load_decoder(args.model)
    print(f"device       {device} ({model.dtype})")
    print(f"gen kwargs   {gen_kwargs}")

    failures = []
    for i, (config, split) in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {config}/{split}", flush=True)
        try:
            path = decode_split(args, model, processor, device, gen_kwargs, config, split)
        except Exception as exc:  # one bad split must not lose the others
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append({"config": config, "split": split, "error": repr(exc)})
        else:
            print(f"\n  wrote {path}")

    if failures:
        log = args.results_dir / "failures.json"
        log.write_text(json.dumps(failures, indent=2))
        print(f"\nFAILED: {len(failures)} of {len(pending)}, logged to {log}", file=sys.stderr)
        return 1

    print(f"\nOK: {len(pending)} split(s). Score with: python -m scripts.score_eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
