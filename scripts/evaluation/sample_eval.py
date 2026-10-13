"""Score a trained model the way inference works: integrate from noise, then compute WER.

This is the number comparable to the leaderboard. The binned metrics logged during
training score a single denoising step from an x_t built out of the target, so at high t
they largely measure the input — the corruption sweep recovers ~75% of tokens at t=0.8
with no model at all.

The default is a sweep over step counts, because the toy validation showed the two things
one might want do not move together: more steps pull the sample onto the data manifold
(fluent, well-formed) while a single step is the model's best per-token guess. Which one
wins on WER is an empirical question per checkpoint, so measure it rather than assume.

    python -m scripts.evaluation.sample_eval --ckpt runs/.../epoch=49.ckpt
    python -m scripts.evaluation.sample_eval --ckpt ... --role test --steps 1 4 16 64
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import train as T
from src.checkpointing import (
    load_components,
    load_manifest,
    load_state_into,
)
from src.data import AudioFeatureDataModule
from src.flow.schedule import TimeReparametrization
from src.flow_mapping import MAP_SAMPLERS, FlowMapping
from src.flow_matching import FlowMatching
from src.metrics import corpus_wer, ids_to_text, score_samples

# The config this machine is configured for, to override a checkpoint's own data.env.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Checkpoint, which split to score, and how many steps to try.

    `a.b=c` tokens are split off as OmegaConf overrides the same way train.py takes them,
    so the data can be pointed somewhere else — `data.env=snellius` above all, since the
    features live on scratch there and the config's default resolves to the repo.
    """
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="Lightning checkpoint to score")
    src.add_argument(
        "--components", type=Path,
        help="a runs/<id>/components directory, as src/checkpoint.py writes. Preferred: "
             "it carries plain state_dicts rather than a pickled module",
    )
    p.add_argument(
        "--variant", default="best", choices=["best", "final"],
        help="which saved weights to score, with --components",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="config to rebuild the data from, for a components directory written before "
             "manifests carried run_config. Defaults to the manifest's own copy",
    )
    p.add_argument("--role", choices=["train", "val", "test"], default="val")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument(
        "--method", nargs="+", default=None,
        help="integrators to score. Default: both for a flow map, where one step against "
             "sixteen is the whole question, and the run's own sampling.method otherwise",
    )
    p.add_argument("--n-utts", type=int, default=None, help="cap the split (default: all)")
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="default: the run's own training batch, capped so the integrator fits",
    )
    p.add_argument(
        "--field-budget-gib", type=float, default=8.0,
        help="GiB the (B, canvas, vocab) fields may occupy; the cap is derived from it",
    )
    p.add_argument("--hard", action="store_true", help="feed back one-hot instead of softmax")
    p.add_argument(
        "--show", type=int, default=25,
        help="reference/hypothesis pairs to print, spread evenly over the split",
    )
    p.add_argument(
        "--examples", type=Path, default=None,
        help="write every reference/hypothesis pair, with its WER, to this TSV",
    )
    args = p.parse_args(flags)
    args.overrides = overrides
    return args


def _config_from_components(args: argparse.Namespace) -> "DictConfig":
    """The run config a components directory was written under.

    Manifests carry the whole resolved config; ones written before that do not, so a
    --config is required for them and the manifest is used to check it matches.
    """
    manifest = load_manifest(args.components)
    if args.config is not None:
        cfg, _ = T.build_config(["--config", str(args.config), *args.overrides])
    elif manifest.get("run_config") is not None:
        cfg = OmegaConf.create(manifest.run_config)
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    else:
        raise SystemExit(
            f"{args.components}/manifest.yaml predates run_config; pass --config with the "
            f"YAML this run was trained from (variant={manifest.get('variant')}, "
            f"size={manifest.get('size')}, mu={manifest.get('mu_size')})"
        )
    # `data.env` says which machine's filesystem to read, which is a fact about where this
    # is running and not about the checkpoint. A run trained on the cluster carries
    # env=snellius, so scoring it on a laptop looks for /scratch-shared and fails with a
    # message telling you to precompute features you already have. Reset unless the caller
    # asked for one explicitly.
    if not any(o.startswith("data.env=") for o in args.overrides) and args.config is None:
        here = T.load_config(T.PROFILES_DIR / "where" / f"{T.detect_where()}.yaml").data.env
        if cfg.data.env != here:
            print(f"data.env {cfg.data.env} -> {here} (this machine, not the checkpoint's)")
            cfg.data.env = here

    # `train.compile` is the same kind of inherited run-time choice, and here it is not
    # merely unhelpful but fatal: a compiled module's state_dict keys all carry an
    # `_orig_mod.` prefix, so loading plain component weights into one fails on every key.
    # It also buys nothing for a single scoring pass, where the ~40 s warmup is pure cost.
    if not any(o.startswith("train.compile=") for o in args.overrides) and cfg.train.compile:
        print("train.compile true -> false (component weights are saved unwrapped)")
        cfg.train.compile = False

    # The manifest knows which vocabulary the weights were trained over, so take it from
    # there rather than from --config. Getting it wrong changes vocab_size by 43x and the
    # weights simply will not fit; making the caller remember it is a trap.
    want_compact = manifest.get("compact_vocab")
    if want_compact is not None and cfg.data.compact_vocab != bool(want_compact):
        print(f"data.compact_vocab {cfg.data.compact_vocab} -> {bool(want_compact)} "
              f"(from the manifest)")
        cfg.data.compact_vocab = bool(want_compact)

    # Sample on the schedule the run settled into, not the prior it started from.
    settled = manifest.get("mu_accuracy_final")
    if settled is not None:
        prior = cfg.schedule.mu_accuracy
        cfg.schedule.mu_accuracy = float(settled)
        print(f"tau from the run's settled mu accuracy {float(settled):.4f}, "
              f"not the prior {prior}")
    for key, want in (("canvas_len", manifest.canvas_len),):
        got = cfg.data[key]
        if got != want:
            raise SystemExit(
                f"data.{key} is {got} but the weights were saved at {want}; the config "
                f"does not describe this run"
            )
    return cfg


def _vocab_hint(cfg: DictConfig) -> int:
    """Vocabulary width, before the data is built — the compact map's size is measured."""
    return 1193 if cfg.data.compact_vocab else 51866


def _build(cfg: DictConfig, dm: AudioFeatureDataModule) -> FlowMatching:
    """An unloaded FlowMatching shaped by `cfg`, ready for weights."""
    return FlowMatching(
        cfg, vocab_size=dm.vocab_size, pad_id=dm.pad_id,
        canvas_len=cfg.data.canvas_len, tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map, time_reparam=T.build_time_reparam(dm, cfg),
    )


class StudentScorer(FlowMapping):
    """A `FlowMapping` whose weights are a trained student's. It samples; it cannot train.

    `FlowMapping` normally loads a teacher and initialises a student from it, which is what
    distillation needs and what `distill.teacher` means everywhere else. Scoring is the one
    case that wants the opposite: the weights are taken as they are, and there is no frozen
    one-time model, so the diagonal term has no target and `map_step` refuses.

    That difference is confined to `load_models`, and it lives here rather than in the class
    because this is the only place that needs it. Everything else — the geometry, the
    probability path, the curve to sample on — comes out of the student's own manifest and
    `tau_<variant>.npz`, which is what makes the directory enough on its own: no teacher has
    to be fetched alongside it.
    """

    def load_models(self) -> tuple[DictConfig, TimeReparametrization | None]:
        """The saved student and its mu, in place of a teacher and a fresh student."""
        distill = self.cfg.distill
        manifest, saved, saved_tau = load_components(
            distill.teacher, variant=distill.get("variant", "best")
        )
        self.check_manifest(manifest)
        if "two_time_denoiser" not in saved:
            raise SystemExit(
                f"{distill.teacher} holds no two_time_denoiser to score; found "
                f"{sorted(saved) or 'nothing'} for variant "
                f"{distill.get('variant', 'best')!r}"
            )
        # No teacher: nothing to distil against, and nothing to freeze.
        self.denoiser = None
        self.mu_nn = saved.get("mu_nn")
        self.two_time_denoiser = saved["two_time_denoiser"]
        return manifest, saved_tau


def _student_from_components(
    args: argparse.Namespace, cfg: DictConfig, dm: AudioFeatureDataModule
) -> StudentScorer:
    """Point `distill.teacher` at the student's own directory and build the scorer."""
    cfg = cfg.copy()
    cfg.distill = OmegaConf.merge(
        cfg.distill,
        OmegaConf.create({"teacher": str(args.components), "variant": args.variant}),
    )
    return StudentScorer(
        cfg,
        vocab_size=dm.vocab_size,
        pad_id=dm.pad_id,
        canvas_len=cfg.data.canvas_len,
        tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map,
    )


def _model_from_components(
    args: argparse.Namespace, cfg: DictConfig, dm: AudioFeatureDataModule
) -> "FlowMatching | FlowMapping":
    """The saved run, rebuilt with its state_dicts loaded into its submodules.

    A distillation run gets a `FlowMapping` instead, keyed on `distill` in its own config
    exactly as `train.build_model` keys on it — a config that names a teacher is a
    distillation config, and a second marker could contradict the first.
    """
    manifest, built, saved = load_components(args.components, variant=args.variant)
    if manifest.vocab_size != dm.vocab_size:
        raise SystemExit(
            f"the weights were saved for vocab {manifest.vocab_size} but this data gives "
            f"{dm.vocab_size}; check data.compact_vocab and the splits"
        )
    if cfg.get("distill") is not None:
        # `StudentScorer.load_models` reads the weights and the curve out of this same
        # directory, so there is nothing left to copy in afterwards.
        return _student_from_components(args, cfg, dm)
    # Geometry and probability path come from the manifest, never from --config: the
    # config supplies the data pipeline, and its model node is whatever that YAML happens
    # to say rather than what these weights were trained as. Taking it from --config means
    # remembering to repeat --size and --mu, and getting it wrong fails deep inside
    # load_state_dict on a missing block.
    cfg = cfg.copy()
    cfg.model = OmegaConf.create(manifest.model)
    cfg.flow = OmegaConf.merge(cfg.flow, OmegaConf.create(manifest.flow))
    model = _build(cfg, dm)
    for name, sub in built.items():
        getattr(model, name).load_state_dict(sub.state_dict(), strict=True)
    # The curve these weights were trained against, exactly, where the run saved one. The
    # mu_accuracy_final rebuild is only the fallback for runs from before it did: it
    # approximates a simulated curve and cannot reproduce a measured one at all.
    if saved is not None:
        model.time_reparam = saved
        print(f"tau from the run's saved tau_{args.variant}.npz: {saved!r}")
    elif model.time_input == "uniform" and not model.time_reparam.is_identity:
        print("warning: no saved tau for a time_input=uniform model; the u fed to the "
              "denoiser follows the rebuilt curve, not the one it trained against")
    return model


def _methods(
    args: argparse.Namespace, cfg: DictConfig, model: "FlowMatching | FlowMapping"
) -> list[str]:
    """Which integrators to score, every table labelled with the one that produced it.

    A flow map defaults to both. One step of it against sixteen of the integrator it was
    distilled from is the question the whole exercise asks, and the two do not agree above
    two steps — so scoring whichever name the config happened to carry would answer half of
    it without saying which half.
    """
    if args.method:
        return [str(m) for m in args.method]
    if isinstance(model, FlowMapping):
        return list(MAP_SAMPLERS)
    return [str(cfg.get("sampling", {}).get("method", "interpolant"))]


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the model from its checkpoint, sample the split, and report WER."""
    args = parse_args(argv)
    if args.components is not None:
        cfg = _config_from_components(args)
    else:
        # The checkpoint carries the config it trained under, which is what makes an old
        # run rescore on its own schedule. Overrides still apply on top — without them the
        # run's own data.subset_n.val would decide how much of the split gets scored, and
        # comparing two runs that scored different amounts is what we are trying to stop.
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = checkpoint["hyper_parameters"]["cfg"]
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.n_utts is not None:
        cfg.data.subset_n[args.role] = args.n_utts
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    # The integrator keeps roughly six (B, canvas_len, vocab_size) fp32 fields alive at
    # once. That is 0.13 GiB per utterance-hundred over the compact vocabulary and 5.5 GiB
    # over Whisper's full one, so a batch that trains happily can be impossible to sample:
    # 128 is routine at 1,193 tokens and OOMs an A100 at 51,866.
    per_utt = cfg.data.canvas_len * _vocab_hint(cfg) * 4 * 6 / 2**30
    cap = max(1, int(args.field_budget_gib / per_utt))
    if cfg.data.batch_size > cap:
        print(f"batch {cfg.data.batch_size} -> {cap}: {per_utt:.3f} GiB per utterance "
              f"of field, against a {args.field_budget_gib:.0f} GiB budget")
        cfg.data.batch_size = cap

    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    if args.components is not None:
        model = _model_from_components(args, cfg, dm)
    else:
        model = _build(cfg, dm)
        for name in ("denoiser", "mu_nn"):
            if getattr(model, name, None) is not None:
                load_state_into(getattr(model, name), checkpoint["state_dict"], name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    loader = {
        "train": dm.train_dataloader,
        "val": dm.val_dataloader,
        "test": dm.test_dataloader,
    }[args.role]()
    # Held on the host and moved one batch at a time: at full latents the val split is
    # ~10 GB in fp16, twice that once cast up, which alone would fill a MIG slice.
    batches = [{"audio": b["audio"], "tokens": b["tokens"]} for b in loader]
    n_utts = sum(b["tokens"].size(0) for b in batches)
    source = args.components or args.ckpt
    print(f"{source}\n  variant {cfg.get('variant')}  role {args.role}  "
          f"{n_utts:,} utterances  vocab {dm.vocab_size:,}\n")

    best = None
    for method in _methods(args, cfg, model):
        print(f"########## method={method} ##########")
        print(f"{'steps':>6} {'WER %':>9} {'token err':>10}")
        for num_steps in args.steps:
            preds, refs = [], []
            for batch in batches:
                audio = batch["audio"].to(device).float()
                preds.append(model.sample(audio, num_steps, hard=args.hard, method=method))
                refs.append(batch["tokens"].to(device))
            pred, ref = torch.cat(preds), torch.cat(refs)
            wer, token_error = score_samples(
                pred, ref, dm.tokenizer, dm.pad_id, dm.vocab_map
            )
            print(f"{num_steps:>6} {wer:9.2f} {token_error:10.4f}")
            if best is None or wer < best[2]:
                best = (method, num_steps, wer, pred, ref)
        print()

    method, steps, wer, pred, ref = best
    print(f"best: {method}, {steps} steps, WER {wer:.2f}%")
    if args.show or args.examples:
        show_examples(pred, ref, dm, args.show, args.examples)
    return 0


def show_examples(
    pred: torch.Tensor,
    ref: torch.Tensor,
    dm: AudioFeatureDataModule,
    n: int,
    out: Path | None = None,
) -> None:
    """Print `n` reference/hypothesis pairs, and optionally write every pair to `out`.

    The `n` are spread evenly over the split rather than taken from its head: LibriSpeech is
    ordered by speaker and chapter, so the first few utterances are one reader. Each carries
    its own WER, from the same normaliser and kaldialign scoring as the corpus number, so a
    bad line can be told from a merely different spelling. `out` is a TSV of the whole split
    (index, WER, reference, hypothesis), the raw text a reader would see.
    """
    hyps = ids_to_text(pred, dm.tokenizer, dm.pad_id, dm.vocab_map)
    refs = ids_to_text(ref, dm.tokenizer, dm.pad_id, dm.vocab_map)
    rows = []
    for i, (r, h) in enumerate(zip(refs, hyps, strict=True)):
        # One pair through the corpus scorer is that utterance's WER; an empty reference has
        # none, as the corpus number drops it too.
        rows.append((i, corpus_wer([r], [h]) if r.strip() else float("nan"), r, h))

    perfect = sum(row[1] == 0 for row in rows)
    print(f"\n{perfect} of {len(rows)} utterances perfect ({100 * perfect / len(rows):.1f}%)")
    picks = sorted({round(k * (len(rows) - 1) / max(n - 1, 1)) for k in range(n)}) if n else []
    for i in picks:
        _, utt_wer, r, h = rows[i]
        print(f"\n  [{i}] WER {utt_wer:5.1f}%")
        print(f"      ref  {r}")
        print(f"      hyp  {h}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            f.write("index\twer\treference\thypothesis\n")
            for i, utt_wer, r, h in rows:
                f.write(f"{i}\t{utt_wer:.1f}\t{r}\t{h}\n")
        print(f"\nwrote all {len(rows)} pairs to {out}")


if __name__ == "__main__":
    sys.exit(main())
