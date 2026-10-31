"""Sampled WER of a finished run on any LibriSpeech splits, for the results table.

`train.evaluate_saved` scores the val and test roles the run trained with, and it rebuilds
the whole datamodule — train split included — to get there. That cannot score splits the
run never saw: for a compact-vocabulary run the vocabulary would be rebuilt from the new
texts, renumbering the ids under the weights, and any reference token outside it would
fail to encode. Here the run's own `vocab_map.pt` is loaded, the model's compact samples
are decoded back to Whisper ids, and the references are the raw transcripts — so a token
the compact model cannot emit is simply an error, which is the fair way to score it.

Everything else is `SamplingProbe.score`: the same chunking and seeded noise, the same
decode, the same normalised corpus WER, under the precision validation ran at. On a split
the run did validate on, the number reproduces its logged one.

    python -m scripts.evaluation.eval_splits --components runs/<id>/components
    python -m scripts.evaluation.eval_splits --components ... \
        --splits clean/validation other/validation clean/test other/test --variants best
"""

import argparse
import sys
import types
from collections.abc import Sequence
from pathlib import Path

from src.constants import data_root_for, feature_dir

import torch
from normalizer import data_utils
from transformers import WhisperTokenizerFast

import train as T
from scripts.evaluation.sample_eval import _config_from_components
from src.checkpointing import load_components, load_vocab_map
from src.data.features import AudioFeatureDataset
from src.data.vocab import VocabMap
from src.metrics.scoring import corpus_wer, ids_to_text

SPLITS = ["clean/validation", "other/validation", "clean/test", "other/test"]


def main(argv: Sequence[str] | None = None) -> int:
    """Score every variant on every split and print one table."""
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=SPLITS, help="config/split pairs")
    p.add_argument("--variants", nargs="+", default=["best", "final"])
    p.add_argument("--out", type=Path, default=None,
                   help="TSV of the results; defaults to <components>/eval_splits.tsv")
    args = p.parse_args(flags)
    args.config, args.overrides = None, overrides

    cfg = _config_from_components(args)
    T.setup_reproducibility(cfg)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        cfg.data.model, language="english", task="transcribe"
    )
    to_full = load_vocab_map(args.components, len(tokenizer))
    vocab_map = VocabMap.from_ids(to_full, len(tokenizer)) if to_full is not None else None
    vocab_size = vocab_map.size if vocab_map is not None else len(tokenizer)
    pad_id = (int(vocab_map.to_compact[tokenizer.pad_token_id]) if vocab_map is not None
              else tokenizer.pad_token_id)
    # build_model reads only these off the datamodule; no split has to be loaded for it.
    sizes = types.SimpleNamespace(vocab_size=vocab_size, pad_id=pad_id, tokenizer=tokenizer,
                                  vocab_map=vocab_map)
    model = T.build_model(sizes, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    probe = model.sampling_probe
    method, num_steps, w = probe.methods[0], probe.step_counts[0], probe.cfg_strengths[0]
    chunk = int(cfg.sampling.get("batch_size") or cfg.data.batch_size)
    bf16 = "bf16" in str(cfg.train.get("precision") or "") and device.type == "cuda"
    root = data_root_for(cfg.data.env, cfg.data.get("data_root"))
    print(f"vocab {vocab_size:,} ({'compact' if vocab_map is not None else 'full'})  "
          f"{method} {num_steps} steps  cfg {w}  chunk {chunk}  bf16 {bf16}  root {root}")

    rows = []
    for variant in args.variants:
        if not (args.components / f"{model.components[0]}_{variant}.pth").exists():
            print(f"no _{variant} components, skipped")
            continue
        _, built, saved_tau = load_components(args.components, variant)
        for name, sub in built.items():
            getattr(model, name).load_state_dict(sub.state_dict())
        if saved_tau is not None:
            model.time_reparam = saved_tau
        for spec in args.splits:
            config, split = spec.split("/")
            ds = AudioFeatureDataset(feature_dir(root, cfg.data.model, config, split),
                                     cfg.data.audio_stride)
            texts = ds.texts
            ids = tokenizer(texts).input_ids
            # The canvas cut every training role applies: a longer transcript cannot be
            # represented, so it is left out rather than scored truncated.
            kept = [i for i, x in enumerate(ids) if len(x) <= cfg.data.canvas_len]
            oov_utts = oov_toks = 0
            if vocab_map is not None:
                for i in kept:
                    miss = sum(int(vocab_map.to_compact[t]) < 0 for t in ids[i])
                    oov_toks += miss
                    oov_utts += miss > 0
            generator = torch.Generator(device=device)
            generator.manual_seed(int(cfg.sampling.get("seed", 0)))
            hyps = []
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=bf16):
                for s in range(0, len(kept), chunk):
                    audio = torch.stack([ds[i]["audio"] for i in kept[s : s + chunk]])
                    pred = model.sample(
                        audio.to(device).float(), num_steps,
                        hard=bool(probe.cfg.get("hard", False)), method=method,
                        generator=generator, grid="live", cfg_strength=w,
                    )
                    hyps += ids_to_text(pred, tokenizer, pad_id, vocab_map)
            # As score_samples: a reference that normalises to nothing is not scored.
            pairs = [(texts[i], h) for i, h in zip(kept, hyps, strict=True)
                     if data_utils.normalizer(texts[i]).strip()]
            wer = corpus_wer([r for r, _ in pairs], [h for _, h in pairs])
            rows.append((variant, spec, wer, len(kept), len(texts), oov_utts, oov_toks))
            print(f"{variant:6s} {spec:18s} WER {wer:6.2f}  ({len(kept)}/{len(texts)} utts"
                  + (f", {oov_utts} with {oov_toks} tokens outside the compact vocab)"
                     if vocab_map is not None else ")"), flush=True)

    out = args.out or args.components / "eval_splits.tsv"
    with open(out, "w") as f:
        f.write("variant\tsplit\twer\tkept\ttotal\toov_utts\toov_tokens\n")
        for r in rows:
            f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
