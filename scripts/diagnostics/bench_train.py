"""Time the real training step on whatever machine this runs on.

Every throughput setting in `configs/base.yaml` was measured on a 6 GB laptop GPU with
6 cores and local NVMe. An A100 node has 40 GB, 16 cores and GPFS, so the batch-size
ceiling, the TF32 trade-off and the data/compute balance can all land differently. This
re-measures them where the training will actually run.

It times `trainer.fit` end to end, so a config that starves the GPU on data shows up as
slow rather than as a good compute number — which is the failure mode that matters on a
shared filesystem.

    python -m scripts.diagnostics.bench_train --config configs/linear.yaml
    python -m scripts.diagnostics.bench_train --config configs/linear.yaml --steps 60
    python -m scripts.diagnostics.bench_train --suite prod --env snellius   # jobs/batch/bench_train.sh

`--suite prod` is the configuration the full-data sweeps actually run — audio_cond at batch
128 under the snellius profile — one change per row, so each line is one decision. Every
row also reports how busy the GPU was during its timed steps, which is what says whether
two runs can share it.
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Sequence

import torch

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
    # Batch and workers both helped on their own and neither was near a limit, so the
    # combination is the untested cell. Peak was 4.44 GiB at batch 256 on a 40 GB card.
    ("batch 128 + workers 16", ["data.batch_size=128", "data.num_workers=16"]),
    ("batch 256 + workers 16", ["data.batch_size=256", "data.num_workers=16"]),
    ("batch 512 + workers 16", ["data.batch_size=512", "data.num_workers=16"]),
    ("batch 256 + workers 16 + compile",
     ["data.batch_size=256", "data.num_workers=16", "train.compile=true"]),
    # Once the fp16 pool freed the dataloader, compile started paying (+33% at batch 256)
    # and inductor began asking for TF32. Both had measured neutral while the step was
    # waiting on data, so the compute-side knobs are worth a second look here.
    ("batch 256 + workers 16 + compile + matmul high",
     ["data.batch_size=256", "data.num_workers=16", "train.compile=true",
      "train.matmul_precision=high"]),
    ("batch 512 + workers 16 + compile",
     ["data.batch_size=512", "data.num_workers=16", "train.compile=true"]),
    # The size ladder, at the settings the rows above settled on. compile is left off:
    # its warmup grows with the model and would land inside the timed window on the deeper
    # geometries. Read these as relative cost per size; compile's gain applies on top.
    *[
        (f"size {name}", ["--size", name, "data.batch_size=256", "data.num_workers=16"])
        for name in ("tiny", "small", "base", "flash", "large")
    ],
    # Canvas width. 96 drops 8.4% of the corpus and train.100 is the longest split, so the
    # loss falls on the training data; 128 keeps 99.75%, 224 keeps everything at 70%
    # padding. Self-attention is quadratic in this and cross-attention linear, so the cost
    # is worth measuring rather than assuming.
    *[
        (f"canvas {c} + small",
         ["--size", "small", f"data.canvas_len={c}",
          "data.batch_size=256", "data.num_workers=16"])
        for c in (96, 128, 224)
    ],
]


# The full-data recipe (jobs/sweeps/mu_reg.sh's control) under the snellius profile, then
# one change per row. Training WER is on in the first row only, as it was in every run so
# far; the profile turns it off. bf16 rather than fp16: the A100 has it natively, needs no
# grad scaler, and DRAX already autocasts its blocks to it.
PROD = ["data.batch_size=128", "train.denoiser_lr=0.002"]
PROD_TRIALS: list[tuple[str, list[str]]] = [
    ("prod as run (train WER on)", [*PROD, "metrics.wer_bins.train=10"]),
    ("prod", PROD),
    ("prod + compile", [*PROD, "train.compile=true"]),
    ("prod + compile + bf16", [*PROD, "train.compile=true", "train.precision=bf16-mixed"]),
    # Informational: fused AdamW cannot run with grad_clip, which every run uses.
    ("prod + compile + fused, no clip",
     [*PROD, "train.compile=true", "train.fused_optimizer=true", "train.grad_clip=null"]),
]



# What a distillation epoch costs, which no cost model knows. `jobs/common/budget.sh` prices
# every job from one `SEC_PER_EPOCH`, measured for audio_cond at the profile's batch with
# compile and bf16 on — and a distillation step differs on both counts: `train.compile` is off
# (map_step indexes by group with `nonzero`, a data-dependent shape), and the step itself does
# more work, since half the batch also runs the frozen teacher and the other half a two-step
# rollout. 1.64x was measured locally at batch 8 with compile off on *both* sides, so it does
# not describe the ratio against a compiled baseline.
#
# The rows isolate the two causes. `distill` is the configuration a sweep would actually run;
# `one-time, compile off` is the same batch without the extra forwards, so the gap between
# them is the objective's cost alone; and the profile's own compiled number is what
# SEC_PER_EPOCH already holds, so the gap to that is what compile is worth here.
#
# `distill + compile` is expected to fail or to recompile every step. It is here to find out
# which, since that is the only thing standing between distillation and the profile default.
DISTILL = ["train.two_time_denoiser_lr=0.00005"]
DISTILL_TRIALS: list[tuple[str, list[str]]] = [
    ("distill", DISTILL),
    ("distill + bf16", [*DISTILL, "train.precision=bf16-mixed"]),
    ("distill + compile", [*DISTILL, "train.compile=true"]),
]


# Marks a trial that feeds one batch, kept on the GPU, back every step (see time_one).
REPLAY = "__replay__"


def gpu_trials(at_batch: int, batches: Sequence[int]) -> list[tuple[str, list[str]]]:
    """How far batch size fills the A100 under the snellius profile, and what the data path costs.

    The ladder comes first because batch is the biggest lever: it amortises per-step launch
    overhead. utts/s flattening out says something stopped scaling. `replay N` then times
    the same step with one batch kept on the GPU and fed back — no loader, no pinning, no
    host-to-device copy — so a replay row well above the real one says the data path, not
    the GPU, is the limit. At stride 1 a batch carries ~3.8 MB of latents per utterance.
    """
    ladder = [(f"batch {b}", [f"data.batch_size={b}"]) for b in batches]
    # Two points, so a gap that grows with the batch (bytes moved) reads differently from
    # a fixed one (per-step overhead).
    replay = [(f"replay {b}", [f"data.batch_size={b}", REPLAY])
              for b in sorted({max(at_batch // 2, 1), at_batch})]
    # The same ladder with the next batch copied to the GPU while the current one trains
    # (data.gpu_prefetch, CudaPrefetcher): how much of the real-vs-replay gap it recovers.
    prefetch = [(f"prefetch {b}", [f"data.batch_size={b}", "data.gpu_prefetch=true"])
                for b in batches]
    return [*ladder, *replay, *prefetch]


SUITES = {
    "laptop": (TRIALS, "configs/linear.yaml"),
    # The recipe, not configs/audio_cond.yaml: it is what cluster runs train (small + micro)
    # and what passes the recipe guard under `--env snellius`.
    "prod": (PROD_TRIALS, "configs/recipes/fulldata.yaml"),
    "gpu": (None, "configs/recipes/fulldata.yaml"),  # built by gpu_trials(--at-batch, --batches)
    # The recipe, not configs/flow_mapping.yaml directly: `--env snellius` goes through the
    # recipe guard in build_config, and the recipe adds nothing but its own name.
    "distill": (DISTILL_TRIALS, "configs/recipes/distill.yaml"),
}


class GpuSampler:
    """GPU utilisation every 0.5 s while it runs, from `nvidia-smi` in the background."""

    def __init__(self) -> None:
        """Not started until `start`."""
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        """Begin sampling; a machine without nvidia-smi just reports nothing."""
        # nvidia-smi numbers GPUs physically and ignores CUDA_VISIBLE_DEVICES, so ask for the
        # one this process was given — index 0 is someone else's GPU in a 2-GPU job.
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                 "-lms", "500", "-i", device],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError:
            self.proc = None

    def stop(self) -> float:
        """Mean utilisation in percent over the sampled window, NaN if none."""
        if self.proc is None:
            return float("nan")
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=5)
        values = [float(v) for v in out.split() if v.strip().replace(".", "").isdigit()]
        return sum(values) / len(values) if values else float("nan")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """What to time and how long to time it for."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--suite", choices=sorted(SUITES), default="laptop",
        help="laptop: the original throughput ladder; prod: the full-data sweep config; "
             "distill: what a flow-map distillation epoch costs, which no cost model knows",
    )
    p.add_argument("--config", default=None, help="defaults to the suite's own config")
    p.add_argument("--at-batch", type=int, default=128,
                   help="gpu suite: the batch the replay trial and --eval-timing run at")
    p.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128, 256, 512],
                   help="gpu suite: the batch ladder (an OOM row is recorded, not fatal)")
    p.add_argument(
        "--eval-timing", action="store_true",
        help="instead of trials, time one full validation pass and one sampling probe at "
             "--at-batch under the profile, which is what an eval cadence is chosen from",
    )
    p.add_argument(
        "--env", choices=["local", "snellius"], default="local",
        help="the machine profile (configs/profiles/where/) the trials run under, which also "
             "sets the data root; the same flag the download and precompute scripts take. "
             "On the cluster this must be snellius",
    )
    p.add_argument("--steps", type=int, default=40, help="timed steps per trial")
    p.add_argument("--warmup", type=int, default=10, help="untimed steps first")
    p.add_argument("--n-utts", type=int, default=2048, help="training subset to read from")
    p.add_argument("--only", nargs="+", default=None, help="run only these trial labels")
    p.add_argument(
        "--set", nargs="+", default=[], metavar="KEY=VALUE",
        help="dotlist overrides applied to every trial, after its own (e.g. the worker "
             "split when two copies share a GPU)",
    )
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
        self.gpu = GpuSampler()

    def on_train_batch_end(self, *_args: object, **_kwargs: object) -> None:
        """Start the clock once warmup is done, then count."""
        if self.started is None:
            self.steps += 1
            if self.steps > self.warmup:
                torch.cuda.synchronize()
                self.started = time.perf_counter()
                self.steps = 0
                self.gpu.start()
            return
        self.steps += 1

    def elapsed(self) -> tuple[float, int, float]:
        """(seconds, steps, mean GPU utilisation %) of steady state."""
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.started
        return seconds, self.steps, self.gpu.stop()


def warm_cache(config: str, args: argparse.Namespace) -> float:
    """Read the whole training subset once, so no trial pays for a cold filesystem.

    On GPFS the first pass over the features is far slower than the rest, and without
    this the trial order decides the winner: each trial inherits a warmer cache than the
    last, which makes per-step time *fall* as batch size rises — an impossible result
    that is easy to mistake for a real one.
    """
    import train as T

    cfg, _ = T.build_config([
        "--config", config, "--offline", "--where", args.env,
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


class ReplayLoader:
    """`steps` iterations of the same on-GPU batch, sized like a DataLoader for Lightning."""

    def __init__(self, batch: dict, steps: int) -> None:
        """Hold the batch; `steps` is what `len` reports to the LR schedule."""
        self.batch, self.steps = batch, steps

    def __len__(self) -> int:
        """Batches per epoch."""
        return self.steps

    def __iter__(self):
        """The same batch, `steps` times."""
        return iter([self.batch] * self.steps)


def time_one(config: str, overrides: list[str], args: argparse.Namespace) -> dict:
    """One trial: build, warm up, then time `--steps` optimizer steps end to end."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback

    import train as T

    total = args.warmup + args.steps
    replay = REPLAY in overrides
    overrides = [o for o in overrides if o != REPLAY]
    cfg, _ = T.build_config([
        "--config", config, "--offline", "--where", args.env,
        f"data.subset_n.train={args.n_utts}",
        f"data.subset_n.val={min(args.n_utts, 64)}",
        f"data.steps_per_epoch={total}",
        "train.num_epochs=1",
        "train.accelerator=gpu",
        "train.checkpoint=false",
        "sampling.enabled=false",
        # Validation is not what we are timing.
        "train.check_val_every_n_epoch=1000000",
        "metrics.wer_bins.train=0", "metrics.wer_bins.val=0",
        *overrides,
        *args.set,
    ])
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = T.FlowMatching(
        cfg, vocab_size=dm.vocab_size, pad_id=dm.pad_id,
        canvas_len=cfg.data.canvas_len, tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map, time_reparam=T.build_time_reparam(dm, cfg),
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
    if replay:
        # One real batch, moved to the GPU once and handed back every step: the step with
        # the data path removed, which is the compute ceiling for this batch.
        batch = next(iter(dm.train_dataloader()))
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        trainer.fit(model, train_dataloaders=ReplayLoader(batch, total))
    else:
        trainer.fit(model, datamodule=dm)
    elapsed, timed_steps, gpu_util = timer.elapsed()
    per_step = elapsed / max(timed_steps, 1)
    return {
        "ms_per_step": per_step * 1000,
        "utts_per_s": cfg.data.batch_size / per_step,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_util": gpu_util,
        "batch": cfg.data.batch_size,
    }


def time_eval(config: str, args: argparse.Namespace) -> dict:
    """Seconds for one validation pass and one sampling probe, as a run pays them.

    Uses the whole validation split and the profile's val metrics (WER decoding included),
    after a short fit so the training graph is compiled. Each is run twice and the second
    is reported: the first also pays compiling the eval-shaped graphs, a one-off per run.
    """
    import lightning.pytorch as pl

    import train as T

    cfg, _ = T.build_config([
        "--config", config, "--offline", "--where", args.env, "--run", "full",
        f"data.batch_size={args.at_batch}",
        f"data.subset_n.train={min(args.n_utts, 4096)}",
        "data.subset_n.val=null",
        f"data.steps_per_epoch={args.warmup}",
        "train.num_epochs=1",
        "train.accelerator=gpu",
        "train.checkpoint=false",
        "sampling.enabled=false",  # validate() must not fire the probe; it is timed alone
        "train.check_val_every_n_epoch=1000000",
        *args.set,
    ])
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = T.FlowMatching(
        cfg, vocab_size=dm.vocab_size, pad_id=dm.pad_id,
        canvas_len=cfg.data.canvas_len, tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map, time_reparam=T.build_time_reparam(dm, cfg),
    )
    trainer = pl.Trainer(
        max_epochs=1, accelerator="gpu", devices=1,
        precision=cfg.train.precision or "32-true",
        gradient_clip_val=cfg.train.grad_clip or 0,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)

    def timed(fn) -> list[float]:
        out = []
        for _ in range(2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append(time.perf_counter() - started)
        return out

    val = timed(lambda: trainer.validate(model, datamodule=dm, verbose=False))
    # Lightning's teardown hands the module back on the CPU after validate(); in a run the
    # probe is called from inside the trainer, with the model still on the GPU.
    model.to("cuda")
    steps = model.sampling_probe.step_counts[0]
    probe = timed(lambda: model.sampling_probe.score(model, dm, "interpolant", num_steps=steps))
    return {"val_s": val, "probe_s": probe, "n_val": len(dm.datasets["val"]), "steps": steps}


def main(argv: Sequence[str] | None = None) -> int:
    """Run every trial and print a table, fastest last."""
    args = parse_args(argv)
    suite, default_config = SUITES[args.suite]
    if suite is None:
        suite = gpu_trials(args.at_batch, args.batches)
    args.config = args.config or default_config
    if args.eval_timing:
        print(f"device: {torch.cuda.get_device_name(0)}   data root: {args.env}")
        r = time_eval(args.config, args)
        print(f"\neval at batch {args.at_batch} under the profile, {r['n_val']} val utts:")
        print(f"  validation pass : {r['val_s'][1]:6.1f} s   (first, with compile: {r['val_s'][0]:.1f} s)")
        print(f"  sampling probe  : {r['probe_s'][1]:6.1f} s   ({r['steps']} steps; first: {r['probe_s'][0]:.1f} s)")
        return 0
    trials = suite
    if args.only:
        # A label that matches nothing is almost always a stale checkout rather than a
        # typo, and silently running the subset that does match burns a whole job to
        # re-measure one row. Name the misses and stop.
        unknown = [name for name in args.only if name not in dict(suite)]
        if unknown:
            print("no such trial: " + ", ".join(repr(u) for u in unknown))
            print("known trials:\n  " + "\n  ".join(label for label, _ in suite))
            return 1
        trials = [t for t in suite if t[0] in args.only]

    print(f"device: {torch.cuda.get_device_name(0)}   data root: {args.env}")
    print(f"timing {args.steps} steps (+{args.warmup} warmup) per trial, "
          f"reading from {args.n_utts} utterances")
    if not args.no_warm:
        warm_cache(args.config, args)
    print(f"{'trial':<34} {'ms/step':>9} {'utts/s':>9} {'peak GiB':>9} {'GPU %':>6}")

    results = []
    for label, overrides in trials:
        try:
            runs = [time_one(args.config, overrides, args) for _ in range(args.repeats)]
            r = min(runs, key=lambda d: d["ms_per_step"])
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<34} {'OOM':>9}")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            # Wide enough to keep the actionable part: the resolver's message names the
            # directories it looked in and the command that would create them.
            first_line = " ".join(str(exc).split("\n")[0].split())
            print(f"{label:<34} FAILED {type(exc).__name__}: {first_line[:120]}")
            if not results:  # a first-trial failure is almost always config, not the trial
                print("  full error:\n    " + str(exc).replace("\n", "\n    "))
                return 1
            continue
        results.append((label, r))
        print(f"{label:<34} {r['ms_per_step']:9.1f} {r['utts_per_s']:9.0f} "
              f"{r['peak_gib']:9.2f} {r['gpu_util']:6.0f}")
        torch.cuda.empty_cache()

    if not results:
        return 1
    best = max(results, key=lambda kv: kv[1]["utts_per_s"])
    # Each suite's first row is its baseline: today's default, or the runs as they were.
    base = next((r for lbl, r in results if lbl == suite[0][0]), None)
    gain = f" ({best[1]['utts_per_s'] / base['utts_per_s']:.2f}x {suite[0][0]})" if base else ""
    print(f"\nfastest: {best[0]} at {best[1]['utts_per_s']:.0f} utts/s{gain}")
    print("Use the winning utts/s to size walltimes: STEPS_PER_SEC = utts_per_s / batch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
