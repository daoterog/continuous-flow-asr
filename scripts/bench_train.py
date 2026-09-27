"""Time the real training step on whatever machine this runs on.

Every throughput setting in `configs/base.yaml` was measured on a 6 GB laptop GPU with
6 cores and local NVMe. An A100 node has 40 GB, 16 cores and GPFS, so the batch-size
ceiling, the TF32 trade-off and the data/compute balance can all land differently. This
re-measures them where the training will actually run.

It times `trainer.fit` end to end, so a config that starves the GPU on data shows up as
slow rather than as a good compute number — which is the failure mode that matters on a
shared filesystem.

    python -m scripts.bench_train --config configs/linear.yaml
    python -m scripts.bench_train --config configs/linear.yaml --steps 60
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# (label, overrides). Ordered so the first row is today's default and everything after
# is a single change from it, which keeps each line readable as one decision.
TRIALS: list[tuple[str, list[str]]] = [
    ("default (batch 32)", []),
    ("batch 64", ["data.batch_size=64"]),
    ("batch 128", ["data.batch_size=128"]),
    ("batch 256", ["data.batch_size=256"]),
    ("batch 64 + matmul high", ["data.batch_size=64", "train.matmul_precision=high"]),
    ("batch 64 + 32-true", ["data.batch_size=64", "train.precision=32-true"]),
    ("batch 64 + fused, no clip",
     ["data.batch_size=64", "train.fused_optimizer=true", "train.grad_clip=null"]),
    ("batch 64 + workers 8", ["data.batch_size=64", "data.num_workers=8"]),
    ("batch 64 + workers 16", ["data.batch_size=64", "data.num_workers=16"]),
    ("batch 64 + compile", ["data.batch_size=64", "train.compile=true"]),
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """What to time and how long to time it for."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/linear.yaml")
    p.add_argument(
        "--env", choices=["local", "snellius"], default="local",
        help="which data root the features live under; the same flag the download and "
             "precompute scripts take. On the cluster this must be snellius, or the "
             "features resolve to the empty data/ inside the repo",
    )
    p.add_argument("--steps", type=int, default=40, help="timed steps per trial")
    p.add_argument("--warmup", type=int, default=10, help="untimed steps first")
    p.add_argument("--n-utts", type=int, default=2048, help="training subset to read from")
    p.add_argument("--only", nargs="+", default=None, help="run only these trial labels")
    p.add_argument(
        "--repeats", type=int, default=1,
        help="time each trial this many times and keep the fastest; 2 guards against a "
             "one-off stall on a shared filesystem",
    )
    p.add_argument(
        "--no-warm", action="store_true",
        help="skip the cache-warming pass (only if the data is already hot)",
    )
    return p.parse_args(argv)


class StepTimer:
    """Times only the steady state: from the end of warmup to the end of the run.

    Timing the whole `fit` would divide one-off setup — dataloader spin-up, weights to
    the GPU, the first batch's allocations — across the few steps being measured, which
    at these step counts dominates. Starting the clock after warmup measures the step.
    """

    def __init__(self, warmup: int) -> None:
        """Count `warmup` steps before starting the clock."""
        self.warmup = warmup
        self.started: float | None = None
        self.steps = 0

    def on_train_batch_end(self, *_args: object, **_kwargs: object) -> None:
        """Start the clock once warmup is done, then count."""
        if self.started is None:
            self.steps += 1
            if self.steps > self.warmup:
                torch.cuda.synchronize()
                self.started = time.perf_counter()
                self.steps = 0
            return
        self.steps += 1

    def elapsed(self) -> tuple[float, int]:
        """(seconds, steps) of steady state."""
        torch.cuda.synchronize()
        return time.perf_counter() - self.started, self.steps


def warm_cache(config: str, args: argparse.Namespace) -> float:
    """Read the whole training subset once, so no trial pays for a cold filesystem.

    On GPFS the first pass over the features is far slower than the rest, and without
    this the trial order decides the winner: each trial inherits a warmer cache than the
    last, which makes per-step time *fall* as batch size rises — an impossible result
    that is easy to mistake for a real one.
    """
    import train as T

    cfg, _ = T.build_config([
        "--config", config, "--offline",
        f"data.env={args.env}",
        f"data.subset_n.train={args.n_utts}",
        "data.subset_n.val=8",
        "data.batch_size=64",
        "data.num_workers=8",
        "schedule.enabled=false",
    ])
    dm = T.build_datamodule(cfg)
    started = time.perf_counter()
    total = 0
    for batch in dm.train_dataloader():
        total += int(batch["audio"].shape[0])
    elapsed = time.perf_counter() - started
    print(f"warming the filesystem cache: {total} utts in {elapsed:.1f}s "
          f"({total / max(elapsed, 1e-9):.0f} utts/s cold)\n")
    return elapsed


def time_one(config: str, overrides: list[str], args: argparse.Namespace) -> dict:
    """One trial: build, warm up, then time `--steps` optimizer steps end to end."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback

    import train as T

    total = args.warmup + args.steps
    cfg, _ = T.build_config([
        "--config", config, "--offline",
        f"data.env={args.env}",
        f"data.subset_n.train={args.n_utts}",
        f"data.subset_n.val={min(args.n_utts, 64)}",
        f"data.steps_per_epoch={total}",
        "train.num_epochs=1",
        "train.accelerator=gpu",
        "train.checkpoint=false",
        "sampling.enabled=false",
        # Validation and the tau sweep are not what we are timing.
        "train.check_val_every_n_epoch=1000000",
        "schedule.sweep_n=256",
        "metrics.wer_bins.train=0", "metrics.wer_bins.val=0",
        *overrides,
    ])
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = T.FlowMatching(
        cfg, vocab_size=dm.vocab_size, pad_id=dm.pad_id, audio_dim=dm.d_model,
        canvas_len=cfg.data.canvas_len, tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map, sample_t_fn=T.build_sample_t_fn(dm, cfg),
    )
    timer = StepTimer(args.warmup)
    hook = type("BenchTimer", (Callback,), {"on_train_batch_end": staticmethod(
        lambda *a, **k: timer.on_train_batch_end()
    )})()
    trainer = pl.Trainer(
        callbacks=[hook],
        max_epochs=1, accelerator="gpu", devices=1,
        precision=cfg.train.precision or "32-true",
        gradient_clip_val=cfg.train.grad_clip or 0,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        limit_val_batches=0,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, datamodule=dm)
    elapsed, timed_steps = timer.elapsed()
    per_step = elapsed / max(timed_steps, 1)
    return {
        "ms_per_step": per_step * 1000,
        "utts_per_s": cfg.data.batch_size / per_step,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "batch": cfg.data.batch_size,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run every trial and print a table, fastest last."""
    args = parse_args(argv)
    trials = TRIALS
    if args.only:
        trials = [t for t in TRIALS if t[0] in args.only]

    print(f"device: {torch.cuda.get_device_name(0)}   data root: {args.env}")
    print(f"timing {args.steps} steps (+{args.warmup} warmup) per trial, "
          f"reading from {args.n_utts} utterances")
    if not args.no_warm:
        warm_cache(args.config, args)
    print(f"{'trial':<30} {'ms/step':>9} {'utts/s':>9} {'peak GiB':>9}")

    results = []
    for label, overrides in trials:
        try:
            runs = [time_one(args.config, overrides, args) for _ in range(args.repeats)]
            r = min(runs, key=lambda d: d["ms_per_step"])
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<30} {'OOM':>9}")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            # Wide enough to keep the actionable part: the resolver's message names the
            # directories it looked in and the command that would create them.
            first_line = " ".join(str(exc).split("\n")[0].split())
            print(f"{label:<30} FAILED {type(exc).__name__}: {first_line[:120]}")
            if not results:  # a first-trial failure is almost always config, not the trial
                print("  full error:\n    " + str(exc).replace("\n", "\n    "))
                return 1
            continue
        results.append((label, r))
        print(f"{label:<30} {r['ms_per_step']:9.1f} {r['utts_per_s']:9.0f} "
              f"{r['peak_gib']:9.2f}")
        torch.cuda.empty_cache()

    if not results:
        return 1
    best = max(results, key=lambda kv: kv[1]["utts_per_s"])
    base = next((r for lbl, r in results if lbl.startswith("default")), None)
    gain = f" ({best[1]['utts_per_s'] / base['utts_per_s']:.2f}x default)" if base else ""
    print(f"\nfastest: {best[0]} at {best[1]['utts_per_s']:.0f} utts/s{gain}")
    print("Use the winning utts/s to size walltimes: STEPS_PER_SEC = utts_per_s / batch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
