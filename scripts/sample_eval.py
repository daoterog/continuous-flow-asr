"""Score a trained model the way inference works: integrate from noise, then compute WER.

This is the number comparable to the leaderboard. The binned metrics logged during
training score a single denoising step from an x_t built out of the target, so at high t
they largely measure the input — the corruption sweep recovers ~75% of tokens at t=0.8
with no model at all.

The default is a sweep over step counts, because the toy validation showed the two things
one might want do not move together: more steps pull the sample onto the data manifold
(fluent, well-formed) while a single step is the model's best per-token guess. Which one
wins on WER is an empirical question per checkpoint, so measure it rather than assume.

    python -m scripts.sample_eval --ckpt runs/.../epoch=49.ckpt
    python -m scripts.sample_eval --ckpt ... --role test --steps 1 4 16 64
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import DictConfig, OmegaConf  # noqa: E402

import train as T  # noqa: E402
from src.checkpoint import (  # noqa: E402
    load_components,
    load_manifest,
    load_state_into,
)
from src.dataset import AudioFeatureDataModule  # noqa: E402
from src.metrics import ids_to_text  # noqa: E402
from src.pl_module import FlowMatching  # noqa: E402
from src.sampler import score_samples  # noqa: E402


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
    p.add_argument("--n-utts", type=int, default=None, help="cap the split (default: all)")
    p.add_argument("--batch-size", type=int, default=None, help="default: the run's own")
    p.add_argument("--hard", action="store_true", help="feed back one-hot instead of softmax")
    p.add_argument("--show", type=int, default=3, help="example transcripts to print")
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


def _build(cfg: DictConfig, dm: AudioFeatureDataModule) -> FlowMatching:
    """An unloaded FlowMatching shaped by `cfg`, ready for weights."""
    return FlowMatching(
        cfg, vocab_size=dm.vocab_size, pad_id=dm.pad_id, audio_dim=dm.d_model,
        canvas_len=cfg.data.canvas_len, tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map, sample_t_fn=T.build_sample_t_fn(dm, cfg),
    )


def _model_from_components(
    args: argparse.Namespace, cfg: DictConfig, dm: AudioFeatureDataModule
) -> FlowMatching:
    """A FlowMatching with the saved state_dicts loaded into its submodules."""
    manifest, built = load_components(args.components, variant=args.variant)
    if manifest.vocab_size != dm.vocab_size:
        raise SystemExit(
            f"the weights were saved for vocab {manifest.vocab_size} but this data gives "
            f"{dm.vocab_size}; check data.compact_vocab and the splits"
        )
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
    return model


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
    batches = [
        {"audio": b["audio"].to(device).float(), "tokens": b["tokens"].to(device)}
        for b in loader
    ]
    n_utts = sum(b["tokens"].size(0) for b in batches)
    source = args.components or args.ckpt
    print(f"{source}\n  variant {cfg.get('variant')}  role {args.role}  "
          f"{n_utts:,} utterances  vocab {dm.vocab_size:,}\n")

    print(f"{'steps':>6} {'WER %':>9} {'token err':>10}")
    best = None
    for num_steps in args.steps:
        preds, refs = [], []
        for batch in batches:
            preds.append(model.sample(batch["audio"], num_steps, hard=args.hard))
            refs.append(batch["tokens"])
        pred, ref = torch.cat(preds), torch.cat(refs)
        wer, token_error = score_samples(
            pred, ref, dm.tokenizer, dm.pad_id, dm.vocab_map
        )
        print(f"{num_steps:>6} {wer:9.2f} {token_error:10.4f}")
        if best is None or wer < best[1]:
            best = (num_steps, wer, pred, ref)

    steps, wer, pred, ref = best
    print(f"\nbest: {steps} steps, WER {wer:.2f}%")
    if args.show:
        hyps = ids_to_text(pred[: args.show], dm.tokenizer, dm.pad_id, dm.vocab_map)
        gold = ids_to_text(ref[: args.show], dm.tokenizer, dm.pad_id, dm.vocab_map)
        print()
        for i, (g, h) in enumerate(zip(gold, hyps, strict=True)):
            print(f"  [{i}] ref  {g[:100]}")
            print(f"      hyp  {h[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
