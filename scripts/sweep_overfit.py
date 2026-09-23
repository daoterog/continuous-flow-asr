"""Overfit sanity check: can the model memorise a handful of utterances?

A model that cannot drive the loss to ~0 on 64 examples it sees hundreds of times has a
bug — in the objective, the interpolant, or the way the audio reaches the denoiser — and
no amount of data will fix it. So regularisation is deliberately switched off (dropout 0,
weight decay 0) and validation runs on *the same* utterances as training: val and train
should sit on top of each other, and both should approach zero.

The sweep runs through submitit's local executor, a few jobs at a time because there is
one GPU. Each job is an ordinary `train.py` configuration, so anything learned here
transfers straight to a real run.

    python -m scripts.sweep_overfit --dry-run          # print the grid and exit
    python -m scripts.sweep_overfit                    # run it
    python -m scripts.sweep_overfit --parallel 2       # two jobs at a time

Results land in wandb under one group, and a summary table prints at the end.
"""

import argparse
import itertools
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import submitit
import yaml

DOTLIST = re.compile(r"^[A-Za-z_][\w.]*=")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Defaults for the axes worth varying. Learning rate is the one that actually decides
# whether memorisation happens; the variant and the schedule are here because a bug in
# the mu path or the reparameterisation would show up as "this one cannot overfit".
# Each is narrowable from the command line — start with one variant, widen once it works.
VARIANTS = ["linear", "audio_cond", "three_term"]
LEARNING_RATES = [1e-3, 3e-3, 1e-2]
SCHEDULES = {"tau": [True], "uniform": [False], "both": [True, False]}


def build_grid(args: argparse.Namespace) -> list[dict]:
    """Every combination of the selected axes, as a list of override dicts."""
    grid = {
        "variant": args.variants,
        "train.denoiser_lr": args.lrs,
        "schedule.enabled": SCHEDULES[args.schedule],
    }
    if args.mu_lrs:
        grid["train.audio_decoder_lr"] = args.mu_lrs
    for axis in args.grid or []:
        key, _, values = axis.partition("=")
        # YAML so `null` becomes None and `1.0` a float, matching the dotlist overrides.
        grid[key] = [yaml.safe_load(v) for v in values.split(",")]
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*grid.values())
    ]


def base_overrides(args: argparse.Namespace) -> list[str]:
    """The settings every job in the sweep shares."""
    return [
        # Point val at the SAME split as train, not base.yaml's held-out one. With the
        # same source, the same cap and the same seed the DataModule draws the identical
        # subset for both, which is what makes this a memorisation test rather than a
        # generalise-from-64-examples test.
        f"data.splits.val.configs=[{','.join(args.train_configs)}]",
        f"data.splits.val.splits=[{','.join(args.train_splits)}]",
        f"data.splits.train.configs=[{','.join(args.train_configs)}]",
        f"data.splits.train.splits=[{','.join(args.train_splits)}]",
        f"data.subset_n.train={args.n_utts}",
        f"data.subset_n.val={args.n_utts}",
        # WER on train too: with val == train it is the same number, but if the identity
        # ever breaks, the two disagreeing is the signal.
        "metrics.wer_bins.train=10",
        "metrics.wer_bins.val=10",
        f"data.batch_size={args.batch_size}",
        "data.num_workers=2",
        f"train.num_epochs={args.epochs}",
        f"train.check_val_every_n_epoch={args.val_every}",
        # Nothing should stop the model memorising.
        "model.denoiser.dropout=0.0",
        "train.weight_decay=0.0",
        f"train.checkpoint={str(args.keep_checkpoints).lower()}",
        "train.precision=16-mixed",
        "train.accelerator=gpu",
        # Warmup would dominate jobs this short.
        "train.compile=false",
        f"schedule.sweep_n={args.sweep_n}",
        # The number that actually matters: integrate from noise and score the result.
        "sampling.enabled=true",
        f"sampling.num_steps={args.sample_steps}",
        f"sampling.every_n_epochs={args.val_every}",
        f"sampling.n_utts={args.n_utts}",
        f"logging.project={args.project}",
        f"logging.group={args.group}",
    ] + ([f"logging.entity={args.entity}"] if args.entity else [])


def as_override(value: object) -> str:
    """Render a grid value the way OmegaConf's dotlist parser reads it back.

    str(None) is "None", which YAML parses as the *string* "None" rather than null — so a
    swept `train.grad_clip=null` arrived at Lightning as text. yaml.safe_dump round-trips
    every scalar correctly.
    """
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("...").strip()


def job_name(combo: dict) -> str:
    """Short label for one grid point."""
    name = f"{combo['variant']}-lr{combo['train.denoiser_lr']:g}"
    if "train.audio_decoder_lr" in combo:
        name += f"-mu{combo['train.audio_decoder_lr']:g}"
    known = {"variant", "train.denoiser_lr", "train.audio_decoder_lr", "schedule.enabled"}
    for key, value in combo.items():
        if key not in known:
            name += f"-{key.rsplit('.', 1)[-1]}={value}"
    return name + f"-{'tau' if combo['schedule.enabled'] else 'uniform'}"


def train_one(config: str, overrides: list[str]) -> dict:
    """Run one configuration and return its final numbers.

    Imported lazily inside the job so the parent process never initialises CUDA, which
    would otherwise be inherited by every forked worker.
    """
    import wandb

    import train as T

    cfg, _ = T.build_config(["--config", config, *overrides])
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = T.FlowMatching(
        cfg,
        vocab_size=dm.vocab_size,
        pad_id=dm.pad_id,
        audio_dim=dm.d_model,
        canvas_len=cfg.data.canvas_len,
        tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map,
        sample_t_fn=T.build_sample_t_fn(dm, cfg),
    )
    logger = T.build_logger(cfg)
    trainer = T.build_trainer(cfg, logger)
    trainer.fit(model, datamodule=dm)

    def metric(name: str) -> float:
        value = trainer.callback_metrics.get(name)
        return float("nan") if value is None else float(value)

    result = {
        "name": cfg.logging.get("run_name", ""),
        "overrides": overrides,
        "train_loss": metric("total_loss/train"),
        "denoiser_loss": metric("denoiser_loss/train_overall"),
        "denoiser_perr": metric("denoiser_perr/train_overall"),
        "train_wer": metric("denoiser_wer/train_overall"),
        "val_wer": metric("denoiser_wer/val_overall"),
        "mu_perr": metric("mu_perr/train_overall"),
        "mu_wer": metric("mu_wer/val_overall"),
        "sampled_wer": metric("sampled/wer"),
        "sampled_perr": metric("sampled/perr"),
        "val_perr": metric("denoiser_perr/val_overall"),
        "wandb_url": getattr(logger.experiment, "url", ""),
    }
    wandb.finish()
    return result


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Sweep size and job shape; `a.b=c` tokens pass through to every job."""
    raw = list(sys.argv[1:] if argv is None else argv)
    # A bare `a.b=c` is a config override for every job. A token directly after an option
    # is that option's value, never an override — `--grid train.grad_clip=1.0,null` would
    # otherwise lose its argument.
    flags, extra = [], []
    for i, token in enumerate(raw):
        after_option = i > 0 and raw[i - 1].startswith("-")
        if not after_option and not token.startswith("-") and DOTLIST.match(token):
            extra.append(token)
        else:
            flags.append(token)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=VARIANTS,
        help="which configs/<name>.yaml to sweep",
    )
    p.add_argument(
        "--lrs", nargs="+", type=float, default=LEARNING_RATES,
        help="denoiser learning rates to try",
    )
    p.add_argument(
        "--mu-lrs", nargs="+", type=float, default=None,
        help="audio-decoder learning rates. Left off, mu is tied to the denoiser's lr; "
             "give values to sweep it as its own axis. Never inherits base.yaml, which "
             "would leave mu at a rate the denoiser sweep has already ruled out.",
    )
    p.add_argument(
        "--schedule", choices=list(SCHEDULES), default="both",
        help="draw t through tau^-1 ('tau'), uniformly ('uniform'), or sweep both",
    )
    p.add_argument("--train-configs", nargs="+", default=["clean"],
                   help="corpus config(s) the memorisation set is drawn from")
    p.add_argument("--train-splits", nargs="+", default=["train.100"],
                   help="split(s) the memorisation set is drawn from; val reuses these")
    p.add_argument("--n-utts", type=int, default=64, help="utterances to memorise")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--val-every", type=int, default=20, help="epochs between validations")
    p.add_argument("--sweep-n", type=int, default=256, help="utterances in the tau sweep")
    p.add_argument("--sample-steps", type=int, default=16, help="integration steps when sampling")
    p.add_argument("--parallel", type=int, default=1, help="jobs at a time on the one GPU")
    p.add_argument("--timeout-min", type=int, default=120)
    p.add_argument("--log-dir", default="runs/sweeps/overfit")
    p.add_argument("--project", default="flow-asr-overfit")
    p.add_argument("--group", default="overfit-sweep")
    p.add_argument("--entity", default=None, help="wandb team; falls back to the config")
    p.add_argument(
        "--grid", action="append", metavar="KEY=V1,V2",
        help="sweep any config key as an extra axis, e.g. --grid train.grad_clip=1.0,null "
             "--grid flow.mu_loss_weight=1.0,0.1. Repeatable; values parse as YAML.",
    )
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="save checkpoints so scripts.sample_eval can probe the result")
    p.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    return p.parse_args(flags), extra


def main(argv: Sequence[str] | None = None) -> int:
    """Build the grid, run it through submitit, and print the summary table."""
    args, extra = parse_args(argv)
    combos = build_grid(args)

    print(f"{len(combos)} jobs, {args.parallel} at a time, {args.epochs} epochs on "
          f"{args.n_utts} utterances")
    for combo in combos:
        print(f"  {job_name(combo)}")
    if args.dry_run:
        print("\n(dry run — nothing submitted)")
        return 0

    executor = submitit.AutoExecutor(folder=args.log_dir, cluster="local")
    executor.update_parameters(timeout_min=args.timeout_min, gpus_per_node=1)

    results = []
    for start in range(0, len(combos), args.parallel):
        wave = combos[start : start + args.parallel]
        jobs = []
        for combo in wave:
            overrides = base_overrides(args) + [
                f"{k}={as_override(v)}" for k, v in combo.items() if k != "variant"
            ]
            overrides += extra
            if "train.audio_decoder_lr" not in combo:
                # Tie mu to the denoiser rather than let it keep base.yaml's value.
                overrides.append(f"train.audio_decoder_lr={combo['train.denoiser_lr']}")
            config = f"configs/{combo['variant']}.yaml"
            jobs.append((combo, executor.submit(train_one, config, overrides)))
        for combo, job in jobs:
            try:
                out = job.result()
                out["label"] = job_name(combo)
                results.append(out)
                print(f"  done {out['label']:30s} loss {out['train_loss']:7.4f}  "
                      f"1-step P_e {out['denoiser_perr']:.4f}  "
                      f"SAMPLED WER {out['sampled_wer']:7.2f}%")
            except Exception as exc:
                print(f"  FAILED {job_name(combo)}: {type(exc).__name__} {exc}")

    if not results:
        print("\nno job completed")
        return 1

    # Ranked by the sampled WER, not the single-step metrics: the binned ones score one
    # denoising step from an x_t built out of the target, so at high t they largely
    # measure the input rather than the model.
    results.sort(key=lambda r: (r["sampled_wer"], r["train_loss"]))
    width = max(len(r["label"]) for r in results)
    print(f"\n{'run':<{width}} {'loss':>8} {'den P_e':>8} {'den WER':>9} "
          f"{'mu P_e':>8} {'mu WER':>9} {'SAMPLED WER':>12}")
    for r in results:
        print(f"{r['label']:<{width}} {r['train_loss']:8.4f} {r['denoiser_perr']:8.4f} "
              f"{r['val_wer']:8.2f}% {r['mu_perr']:8.4f} {r['mu_wer']:8.2f}% "
              f"{r['sampled_wer']:11.2f}%")

    best = results[0]
    print(f"\nbest by sampled WER: {best['label']}  ({best['sampled_wer']:.2f}%)")
    print("A model that has memorised these utterances generates them from noise: sampled "
          "WER ~ 0.\nA low 1-step P_e with a high sampled WER means it can denoise a "
          "mostly-correct input\nbut cannot generate — which is the failure the binned "
          "metrics alone would hide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
