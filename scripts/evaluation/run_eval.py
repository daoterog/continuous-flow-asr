"""Transcribe Open ASR Leaderboard eval sets and write result manifests.

Two sources, selected with ``--source``:

* **audio** (default) — runs the full encoder-decoder on the audio, the way the leaderboard
  does. This is the path for every dataset, and the only one that yields a comparable RTFx.
* **latents** — decodes cached Whisper encoder latents from ``scripts.data.precompute_whisper``
  via ``generate(encoder_outputs=...)``, skipping the encoder. This reproduced the published
  LibriSpeech numbers and is kept as the regression control: if it drifts, the decode
  changed. It requires **untrimmed** latents; Whisper's decoder cross-attends over all 1500
  frames and emits repeated ``<|en|>`` from a trimmed cache, so a trimmed one is refused
  rather than silently scored.

Predictions and references are written **raw**; the leaderboard's scorer normalizes at
scoring time. Score with ``python -m scripts.evaluation.score_eval``.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from scripts.data.download_librispeech import LOCAL_DATA_ROOT, SNELLIUS_DATA_ROOT

if TYPE_CHECKING:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = REPO_ROOT / "open_asr_leaderboard"

DEFAULT_MODEL = "openai/whisper-large-v3"
EVAL_REPO = "hf-audio/open-asr-leaderboard"
CHUNKED_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"

# key -> (dataset_path, dataset_name, split). score_results derives its CSV column from the
# manifest filename, which embeds all three, so these must yield the substrings it looks
# for: "ami_cleaned_test", "librispeech_test.clean", "earnings22_cleaned_aa_chunked_test".
EVAL_DATASETS: dict[str, tuple[str, str, str]] = {
    "ami_cleaned": (EVAL_REPO, "ami_cleaned", "test"),
    "gigaspeech_cleaned": (EVAL_REPO, "gigaspeech_cleaned", "test"),
    "spgispeech": (EVAL_REPO, "spgispeech", "test"),
    "voxpopuli_cleaned_aa": (EVAL_REPO, "voxpopuli_cleaned_aa", "test"),
    "librispeech_test.clean": (EVAL_REPO, "librispeech", "test.clean"),
    "librispeech_test.other": (EVAL_REPO, "librispeech", "test.other"),
    "earnings22_cleaned_aa_chunked": (CHUNKED_REPO, "earnings22_cleaned_aa_chunked", "test"),
}

# Latent path only: the cache is keyed by openslr's (config, split), which must be relabelled
# to the leaderboard's names so the scorer recognises the manifest.
LATENT_NAMES = {
    ("clean", "test"): ("librispeech", "test.clean"),
    ("other", "test"): ("librispeech", "test.other"),
    ("clean", "validation"): ("librispeech", "validation.clean"),
    ("other", "validation"): ("librispeech", "validation.other"),
}
LATENT_DATASET_PATH = "openslr/librispeech_asr"


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
    p.add_argument("--source", choices=["audio", "latents"], default="audio")
    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--data-root", type=Path, default=None, help="overrides --env")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=list(EVAL_DATASETS),
        default=list(EVAL_DATASETS),
        help="audio source only",
    )
    p.add_argument("--configs", nargs="+", default=["clean", "other"], help="latents only")
    p.add_argument("--splits", nargs="+", default=["test"], help="latents only")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="64 matches the leaderboard's run_whisper.sh; RTFx varies with this",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="untimed batches before measuring, so RTFx excludes CUDA autotuning",
    )
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

    # An empty model still runs, writing manifests named MODEL__DATASET_... that the
    # scorer cannot attribute to a checkpoint.
    if not args.model:
        p.error("--model must not be empty")
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


def manifest_name(
    model: str, dataset_path: str, dataset: str, split: str, part: int | None = None
) -> str:
    """Reproduce the leaderboard's manifest filename, which its scorer parses.

    A sharded run writes ``.partN`` files instead; score_eval concatenates them back into
    this canonical name before scoring, since score_results would otherwise treat each part
    as a separate dataset and average them wrongly.
    """
    stem = (
        f"MODEL_{model.replace('/', '-')}"
        f"_DATASET_{dataset_path.replace('/', '-')}_{dataset}_{split}"
    )
    return f"{stem}.part{part}.jsonl" if part is not None else f"{stem}.jsonl"


def count_utterances(keys: list[str]) -> int:
    """Total utterances across the selected datasets, from the dataset card.

    Reads only card metadata, so it needs neither the audio downloaded nor a model loaded
    — submit/eval.sh uses it to size the sbatch --time before anything is fetched.
    """
    from huggingface_hub import HfApi

    counts: dict[str, int] = {}
    for cfg in HfApi().dataset_info(EVAL_REPO).cardData["dataset_info"]:
        for split in cfg["splits"]:
            counts[f"{cfg['config_name']}_{split['name']}"] = split["num_examples"]

    total = 0
    for key in keys:
        _path, name, split = EVAL_DATASETS[key]
        # The chunked set is in its own repo, so its size is not on this card.
        total += 341 if key == "earnings22_cleaned_aa_chunked" else counts.get(
            f"{name}_{split}", 0
        )
    return total


def load_model(model_id: str) -> tuple[Any, Any, "torch.device", dict]:
    """Load the model and processor, with generation settings matching the leaderboard."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(model_id)
    model = (
        WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
    )

    gen_kwargs: dict = {}
    if getattr(model.generation_config, "is_multilingual", False):
        gen_kwargs["language"] = "en"
        gen_kwargs["task"] = "transcribe"
    # Cleared to match the leaderboard's setup exactly; these change the decode.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    return model, processor, device, gen_kwargs


def write_manifest(path: Path, rows: list[dict]) -> str:
    """Write a result manifest in the leaderboard's JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def decode_from_audio(
    args: argparse.Namespace,
    model: "WhisperForConditionalGeneration",
    processor: "WhisperProcessor",
    device: "torch.device",
    gen_kwargs: dict,
    key: str,
) -> str:
    """Run the full encoder-decoder over one dataset and write its manifest."""
    import torch
    from normalizer import data_utils

    dataset_path, dataset_name, split = EVAL_DATASETS[key]
    sampling_rate = processor.feature_extractor.sampling_rate

    # load_data handles both the plain configs and the chunked earnings22, whose chunks it
    # joins to their parent call transcripts. Safe as a cache hit because
    # scripts.data.download_eval_sets has already fetched the whole config.
    dataset = data_utils.load_data(
        SimpleNamespace(
            dataset_path=dataset_path, dataset=dataset_name, split=split, streaming=False
        )
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate)
    if args.num_shards > 1:
        # contiguous=False interleaves, so each task gets a similar mix of utterance
        # lengths and the tasks finish together.
        dataset = dataset.shard(
            num_shards=args.num_shards, index=args.shard_id, contiguous=False
        )

    is_chunked = data_utils.is_chunked_dataset(dataset_path)

    def benchmark(batch: dict) -> dict:
        audios = [a["array"] for a in batch["audio"]]
        minibatch_size = len(audios)
        rate = batch["audio"][0]["sampling_rate"]
        batch["audio_length_s"] = [len(a) / rate for a in audios]
        # The full id, not the leaderboard's basenamed version. spgispeech ids look like
        # "<hash>/117.wav", and basenaming collapses 1035 distinct utterances into 253
        # names — which makes coverage unverifiable and looks like shard overlap. The
        # field is not used for scoring, so keeping it unique costs nothing.
        ids = batch.get("id")
        batch["audio_filepath"] = (
            [str(i) for i in ids]
            if isinstance(ids, list) and len(ids) == minibatch_size
            else data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)
        )

        # RTFx methodology, matching open_asr_leaderboard/transformers/run_eval.py:172-353:
        # cuda events (not perf_counter) around preprocessing + generate + decode, so the
        # measured span is the same one the published figures were taken over.
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        inputs = processor(audios, sampling_rate=rate, return_tensors="pt")
        inputs = inputs.to(device, model.dtype)
        # Whisper's extractor sets return_attention_mask=False, so generate has nothing to
        # infer a mask from and warns. All-ones is the correct mask, not just a silencer:
        # Whisper attends over the whole padded 30 s window (the reason trimmed latents
        # decode to garbage), and its encoder ignores this argument regardless.
        features = inputs["input_features"]
        mask = torch.ones(
            (features.shape[0], features.shape[-1]), dtype=torch.long, device=device
        )
        with torch.no_grad():
            pred_ids = model.generate(**inputs, attention_mask=mask, **gen_kwargs)
        pred_text = processor.batch_decode(
            pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            runtime = start_event.elapsed_time(end_event) / 1000.0
        else:
            runtime = float("nan")

        # Per-sample, so the times sum back to wall time across the split.
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_text
        batch["references"] = batch["original_text"]
        return batch

    # Warmup calls benchmark directly on slices rather than through .map. Its results are
    # discarded by definition, and .map would Arrow-serialize every warmed batch —
    # including the decoded waveforms, since there is no remove_columns here — for nothing.
    if args.warmup_steps and device.type == "cuda":
        for step in range(args.warmup_steps):
            lo = step * args.batch_size
            if lo >= len(dataset):
                break
            benchmark(dataset[lo : min(lo + args.batch_size, len(dataset))])

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
        # Bounds the Arrow writer buffer; the default 1000 rows holds far more in memory
        # than the batch itself.
        writer_batch_size=args.batch_size,
    )

    keys = [
        "audio_length_s",
        "transcription_time_s",
        "predictions",
        "references",
        "audio_filepath",
    ]
    if is_chunked:
        keys += data_utils.CHUNK_METADATA_KEYS
    collected: dict[str, list] = {k: [] for k in keys}
    for i, row in enumerate(dataset):
        for k in keys:
            collected[k].append(row[k])
        if i % 200 == 0:
            print(f"    {i}/{len(dataset)}", end="\r", flush=True)

    rows = []
    for i in range(len(collected["references"])):
        row = {
            "audio_filepath": collected["audio_filepath"][i] or f"sample_{i}",
            "duration": collected["audio_length_s"][i],
            "time": collected["transcription_time_s"][i],
            "text": collected["references"][i],
            "pred_text": collected["predictions"][i],
        }
        if is_chunked:
            # score_results calls merge_chunked_manifest, which concatenates a call's
            # chunks before scoring against the call-level reference.
            for k in data_utils.CHUNK_METADATA_KEYS:
                row[k] = collected[k][i]
        rows.append(row)

    part = args.shard_id if args.num_shards > 1 else None
    path = args.results_dir / manifest_name(
        args.model, dataset_path, dataset_name, split, part
    )
    return write_manifest(path, rows)


def decode_from_latents(
    args: argparse.Namespace,
    model: "WhisperForConditionalGeneration",
    processor: "WhisperProcessor",
    device: "torch.device",
    gen_kwargs: dict,
    config: str,
    split: str,
) -> str:
    """Decode one cached split of latents and write its manifest."""
    import numpy as np
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    src = feature_dir(args.data_root, args.model, config, split)
    indexes = sorted(src.glob("*.json"))
    if not indexes:
        raise FileNotFoundError(
            f"no cached features under {src} — run scripts.data.precompute_whisper "
            f"--configs {config} --splits {split} --no-trim"
        )

    refs: list[str] = []
    preds: list[str] = []
    utt_ids: list[str] = []
    for index_path in indexes:
        meta = json.loads(index_path.read_text())
        if meta["trimmed"]:
            raise ValueError(
                f"{index_path} holds trimmed latents, which Whisper's decoder cannot "
                "decode (it needs all 1500 frames). Re-run precompute with --no-trim."
            )
        feats = np.load(index_path.with_suffix(".npy"), mmap_mode="r")
        utts = meta["utts"][: args.limit] if args.limit else meta["utts"]
        utts = utts[args.shard_id :: args.num_shards]

        for start in range(0, len(utts), args.batch_size):
            chunk = utts[start : start + args.batch_size]
            batch = np.stack([feats[u["offset"] : u["offset"] + u["length"]] for u in chunk])
            hidden = torch.from_numpy(batch).to(device, dtype=model.dtype)
            # Whisper encoder output is always a full 1500 frames, so the mask is all ones;
            # passing it stops generate warning that it cannot infer one.
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=device)
            with torch.no_grad():
                ids = model.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=mask,
                    **gen_kwargs,
                )
            preds.extend(
                processor.batch_decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            )
            refs.extend(u["text"] for u in chunk)
            utt_ids.extend(u["id"] for u in chunk)
            print(f"    {len(preds)}/{len(utts)}", end="\r", flush=True)

    dataset_name, lb_split = LATENT_NAMES.get((config, split), ("librispeech", split))
    # duration/time omitted: the encoder ran in a separate offline job, so an RTFx here
    # would describe the decoder alone. score_results skips RTFx when they are absent.
    rows = [
        {
            "audio_filepath": utt_id,
            "duration": None,
            "time": None,
            "text": ref,
            "pred_text": pred,
        }
        for utt_id, ref, pred in zip(utt_ids, refs, preds, strict=True)
    ]
    part = args.shard_id if args.num_shards > 1 else None
    path = args.results_dir / manifest_name(
        args.model, LATENT_DATASET_PATH, dataset_name, lb_split, part
    )
    return write_manifest(path, rows)


def plan_units(args: argparse.Namespace) -> list[tuple]:
    """Return the (label, out_path, work) units this run will process."""
    part = args.shard_id if args.num_shards > 1 else None
    units = []
    if args.source == "audio":
        for key in args.datasets:
            path, name, split = EVAL_DATASETS[key]
            out = args.results_dir / manifest_name(args.model, path, name, split, part)
            units.append((key, out, (key,)))
    else:
        for config in args.configs:
            for split in args.splits:
                if not feature_dir(args.data_root, args.model, config, split).is_dir():
                    continue
                name, lb_split = LATENT_NAMES.get((config, split), ("librispeech", split))
                out = args.results_dir / manifest_name(
                    args.model, LATENT_DATASET_PATH, name, lb_split, part
                )
                units.append((f"{config}/{split}", out, (config, split)))
    return units


def main(argv: list[str] | None = None) -> int:
    """Transcribe the selected datasets, returning a process exit code."""
    args = parse_args(argv)
    # Before any transformers/hub import: the checkpoint is ~3 GB and would otherwise land
    # in $HOME/.cache/huggingface, which is quota'd on Snellius.
    os.environ["HF_HOME"] = str(args.data_root)
    add_leaderboard_to_path()

    units = plan_units(args)
    if not units:
        print("error: nothing matched the selection", file=sys.stderr)
        return 1
    pending = [u for u in units if args.overwrite or not u[1].is_file()]

    print(f"model        {args.model}")
    print(f"source       {args.source}")
    print(f"data root    {args.data_root}")
    print(f"units        {len(units)} -> {len(pending)} pending")
    print(f"results dir  {args.results_dir}")
    print(f"batch size   {args.batch_size}")
    if args.source == "audio":
        print(f"warmup       {args.warmup_steps} step(s)")
    if args.limit:
        print(f"limit        {args.limit} utt(s) per split  (NOT a full run)")

    if args.dry_run:
        if args.source == "audio":
            # Machine-readable; submit/eval.sh reads this to size the sbatch --time.
            print(f"utterances   {count_utterances([u[0] for u in units])}")
        for label, out, _ in units:
            print(f"  {label} -> {out.name}")
        return 0
    if not pending:
        print("nothing to do")
        return 0

    model, processor, device, gen_kwargs = load_model(args.model)
    print(f"device       {device} ({model.dtype})")
    print(f"gen kwargs   {gen_kwargs}")

    failures = []
    for i, (label, _out, work) in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {label}", flush=True)
        try:
            if args.source == "audio":
                path = decode_from_audio(args, model, processor, device, gen_kwargs, *work)
            else:
                path = decode_from_latents(args, model, processor, device, gen_kwargs, *work)
        except Exception as exc:  # one bad dataset must not lose the others
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append({"unit": label, "error": repr(exc)})
        else:
            print(f"\n  wrote {path}")

    if failures:
        # pending units may all have failed before results_dir was created.
        args.results_dir.mkdir(parents=True, exist_ok=True)
        # Per task: all array tasks share results_dir, so a single failures.json means the
        # last task to fail silently overwrites every earlier one.
        log = args.results_dir / f"eval_failures_{args.shard_id}.json"
        log.write_text(json.dumps(failures, indent=2))
        print(
            f"\nFAILED: {len(failures)} of {len(pending)}, logged to {log}", file=sys.stderr
        )
        return 1

    print(f"\nOK: {len(pending)} unit(s). Score with: python -m scripts.evaluation.score_eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
