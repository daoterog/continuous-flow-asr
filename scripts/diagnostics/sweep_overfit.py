"""Overfit sanity check: can the model memorise a handful of utterances?

A model that cannot drive the loss to ~0 on 64 examples it sees hundreds of times has a
bug — in the objective, the interpolant, or the way the audio reaches the denoiser — and
no amount of data will fix it. So regularisation is deliberately switched off (dropout 0,
weight decay 0) and validation runs on *the same* utterances as training: val and train
should sit on top of each other, and both should approach zero.

The sweep runs through submitit's local executor, a few jobs at a time because there is
one GPU. Each job is an ordinary `train.py` configuration, so anything learned here
transfers straight to a real run.

    python -m scripts.diagnostics.sweep_overfit --dry-run          # print the grid and exit
    python -m scripts.diagnostics.sweep_overfit                    # run it
    python -m scripts.diagnostics.sweep_overfit --parallel 2       # two jobs at a time

Results land in wandb under one group, and a summary table prints at the end.
"""

import argparse
import itertools
import re
import sys
from collections.abc import Sequence

import submitit
import yaml

DOTLIST = re.compile(r"^[A-Za-z_][\w.]*=")


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
    if len(args.n_utts) > 1:
        grid["n_utts"] = args.n_utts
    for axis in args.grid or []:
        key, _, values = axis.partition("=")
        # YAML so `null` becomes None and `1.0` a float, matching the dotlist overrides.
        grid[key] = [yaml.safe_load(v) for v in values.split(",")]
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*grid.values())
    ]


def epochs_for(args: argparse.Namespace, n_utts: int) -> int:
    """Epochs for one memorisation-set size.

    With --max-steps, epochs are derived so every size gets the same number of optimizer
    steps. At fixed epochs a bigger set silently gets more steps (n/batch per epoch), so
    set size and training budget move together and neither can be attributed.
    """
    if not args.max_steps:
        return args.epochs
    # With an explicit steps_per_epoch the epoch no longer depends on the set size, so
    # every job gets the same epochs AND the same steps — and far fewer epoch boundaries,
    # which cost ~100ms each regardless of how much work the epoch contains.
    steps_per_epoch = args.steps_per_epoch or max(1, n_utts // args.batch_size)
    return max(1, round(args.max_steps / steps_per_epoch))


def val_every_for(args: argparse.Namespace, n_utts: int) -> int:
    """Epochs between validations, so every job validates the same NUMBER of times.

    A fixed epoch interval scales badly once epochs are derived from a step budget: at
    n=64 a 25.6k-step run is 6400 epochs, and validating every 20 would sample 320 times
    — each sample being num_steps forward passes over the whole slice, which can cost
    more than the training it is measuring.
    """
    if not args.val_times:
        return args.val_every
    return max(1, epochs_for(args, n_utts) // args.val_times)


def base_overrides(args: argparse.Namespace, n_utts: int) -> list[str]:
    """The settings every job in the sweep shares, for one memorisation-set size."""
    return [
        # Point val at the SAME split as train, not base.yaml's held-out one. With the
        # same source, the same cap and the same seed the DataModule draws the identical
        # subset for both, which is what makes this a memorisation test rather than a
        # generalise-from-64-examples test.
        f"data.splits.val.configs=[{','.join(args.train_configs)}]",
        f"data.splits.val.splits=[{','.join(args.train_splits)}]",
        f"data.splits.train.configs=[{','.join(args.train_configs)}]",
        f"data.splits.train.splits=[{','.join(args.train_splits)}]",
        f"data.subset_n.train={n_utts}",
        f"data.subset_n.val={n_utts}",
        # WER on train too: with val == train it is the same number, but if the identity
        # ever breaks, the two disagreeing is the signal.
        "metrics.wer_bins.train=10",
        "metrics.wer_bins.val=10",
        f"data.batch_size={args.batch_size}",
        "data.num_workers=2",
    ] + ([f"data.steps_per_epoch={args.steps_per_epoch}"] if args.steps_per_epoch else []) + [
        f"train.num_epochs={epochs_for(args, n_utts)}",
        f"train.check_val_every_n_epoch={val_every_for(args, n_utts)}",
        # Nothing should stop the model memorising.
        "model.denoiser.dropout=0.0",
        "train.weight_decay=0.0",
        f"train.checkpoint={str(args.keep_checkpoints).lower()}",
        "train.precision=16-mixed",
        "train.accelerator=gpu",
        # Compile pays for itself once a job runs for minutes rather than seconds: the
        # bottleneck at these batch sizes is kernel-launch overhead, which is what it
        # removes. Warmup is a one-off ~40-70s.
        f"train.compile={str(args.compile).lower()}",
        # The number that actually matters: integrate from noise and score the result.
        "sampling.enabled=true",
        f"sampling.num_steps={args.sample_steps}",
        f"sampling.every_n_epochs={val_every_for(args, n_utts)}",
        f"sampling.n_utts={min(n_utts, 128)}",
        *(
            [
                f"train.early_stop_on={args.stop_on}",
                f"train.early_stop_threshold={args.stop_at}",
            ]
            if args.stop_at is not None
            else []
        ),
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
    if "n_utts" in combo:
        name += f"-n{combo['n_utts']}"
        known.add("n_utts")
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
        canvas_len=cfg.data.canvas_len,
        tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map,
        time_reparam=T.build_time_reparam(dm, cfg),
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
        # What the interpolant already carried, against what the denoiser made of it.
        "input_perr": metric("input_perr/train_overall"),
        "input_perr_low_t": metric("input_perr/train_low_t"),
        "denoiser_perr_low_t": metric("denoiser_perr/train_low_t"),
        "mu_wer": metric("mu_wer/val_overall"),
        "sampled_wer": metric("sampled/wer"),
        "stopped_epoch": trainer.current_epoch,
        "stopped_step": int(trainer.global_step),
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
    p.add_argument("--n-utts", type=int, nargs="+", default=[64],
                   help="utterances to memorise; several values sweep it as an axis, "
                        "which is how to make the task hard enough for mu to matter")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help="make an epoch this many steps by sampling with replacement; on small sets "
             "this removes most of the per-epoch fixed cost",
    )
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="equalise optimizer steps across --n-utts by deriving epochs per size; "
             "use this whenever n_utts is swept, or set size and budget are confounded",
    )
    p.add_argument("--val-every", type=int, default=20, help="epochs between validations")
    p.add_argument(
        "--val-times", type=int, default=None,
        help="validate this many times per run instead of every --val-every epochs; "
             "keeps scoring cost constant when epochs are derived from --max-steps",
    )
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
    p.add_argument(
        "--stop-at", type=float, default=None, metavar="WER",
        help="stop a job once --stop-on reaches this value, e.g. --stop-at 0.5. A job "
             "that has memorised the set has nothing left to learn; one that has not "
             "still runs the full budget, so the sweep becomes 'steps to reach the "
             "target' rather than 'score at a fixed budget'.",
    )
    p.add_argument("--stop-on", default="sampled/wer", help="metric --stop-at watches")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile both models; worth it whenever a job runs minutes")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="save checkpoints so scripts.evaluation.sample_eval can probe the result")
    p.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    return p.parse_args(flags), extra


def main(argv: Sequence[str] | None = None) -> int:
    """Build the grid, run it through submitit, and print the summary table."""
    args, extra = parse_args(argv)
    combos = build_grid(args)

    print(f"{len(combos)} jobs, {args.parallel} at a time, on {args.n_utts} utterance(s)")
    for combo in combos:
        n = combo.get("n_utts", args.n_utts[0])
        epochs = epochs_for(args, n)
        per_epoch = args.steps_per_epoch or max(1, n // args.batch_size)
        print(f"  {job_name(combo):<44} {epochs:>6} epochs x {per_epoch:>4}"
              f" steps = {epochs * per_epoch:>7,} steps")
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
            n_utts = combo.get("n_utts", args.n_utts[0])
            overrides = base_overrides(args, n_utts) + [
                f"{k}={as_override(v)}"
                for k, v in combo.items()
                if k not in ("variant", "n_utts")
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
                      f"SAMPLED WER {out['sampled_wer']:7.2f}%  "
                      f"@{out.get('stopped_step', 0):,} steps")
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
    print(f"\n{'run':<{width}} {'in P_e':>8} {'den P_e':>8} {'mu P_e':>8} "
          f"{'in@lowT':>8} {'den@lowT':>9} {'den WER':>9} {'SAMPLED':>9} {'steps':>9}")
    for r in results:
        print(f"{r['label']:<{width}} {r['input_perr']:8.4f} {r['denoiser_perr']:8.4f} "
              f"{r['mu_perr']:8.4f} {r['input_perr_low_t']:8.4f} "
              f"{r['denoiser_perr_low_t']:9.4f} {r['val_wer']:8.2f}% "
              f"{r['sampled_wer']:8.2f}% {r.get('stopped_step', 0):9,}")
    print("\n  in P_e  = token error of argmax(x_t), the interpolant with no model in it")
    print("  den P_e = the denoiser's, from that same input. The gap is what it added.")
    print("  @lowT   = the same two in the lowest populated t bin, where x_t is nearly")
    print("            pure noise and mu is the only signal the denoiser has.")

    best = results[0]
    print(f"\nbest by sampled WER: {best['label']}  ({best['sampled_wer']:.2f}%)")
    print("A model that has memorised these utterances generates them from noise: sampled "
          "WER ~ 0.\nA low 1-step P_e with a high sampled WER means it can denoise a "
          "mostly-correct input\nbut cannot generate — which is the failure the binned "
          "metrics alone would hide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
