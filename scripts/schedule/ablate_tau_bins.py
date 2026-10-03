"""How many grid points tau^-1 actually needs, and where more stop buying anything.

Historical: the question this answers no longer arises on the path it was written for.
`SimulatedTau` integrates its curve (src/flow/recovery.py) instead of sweeping it, so
there is no sampling noise to trade resolution against and `schedule.num_bins` now applies
only to the measured curve. Kept because the measurement stands on its own, and because it
is what the EXPERIMENTS.md rows on bin count refer to.

The sweep it ablates: the P_e curve is evaluated at `num_bins + 1` values of t, and tau^-1
interpolates between whichever of those survive the flat-run cull. Too few and the map is a
coarse staircase; too many and the sweep costs more for a curve that is already smooth.

The comparison has to separate two error sources that move in opposite directions:

  * discretisation — the grid is too coarse to represent the curve;
  * sampling noise — the sweep estimates recovery from a finite batch, so two runs of the
    same grid disagree anyway.

So the sweep runs ONCE per seed at high resolution and every candidate grid is a
subsample of it. Discretisation is then measured against that seed's own full-resolution
answer, with nothing else changing. The noise floor is the disagreement between two seeds
at full resolution. More bins are worth having only while the first is above the second.

Reported as a shift in the t that training would draw, in t units, which is the thing that
actually matters downstream.

    python -m scripts.schedule.ablate_tau_bins
    python -m scripts.schedule.ablate_tau_bins --device cuda --n-utts 1024 --reference 1600
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.flow.schedule import make_tau_inverse, sweep_corruption

# Quantiles of the drawn t, so the summary says where the schedule moved, not just by how
# much. q50 is the bulk; q01 is the low-t tail, which is the part a coarse grid distorts.
QUANTILES = (1, 10, 50, 90, 99)


def sweep(
    tokens: torch.Tensor, vocab_size: int, n_bins: int, seed: int,
    mu_accuracy: float = 0.0, gamma: bool = False,
) -> np.ndarray:
    """Recovery on `linspace(0, 1, n_bins + 1)`, under a fixed seed.

    `gamma` adds the audio-conditioned term with mu standing in at `mu_accuracy`, which is
    what a refreshed simulated curve actually looks like once mu has learned something.
    """
    torch.manual_seed(seed)
    ts = np.linspace(0.0, 1.0, n_bins + 1)
    return np.asarray(sweep_corruption(
        tokens, vocab_size, ts.tolist(),
        gamma_fn=(lambda t: 1.0 - t) if gamma else None,
        mu_accuracy=mu_accuracy,
    ))


def drawn_t(ts: np.ndarray, recovery: np.ndarray, vocab_size: int, u: np.ndarray) -> np.ndarray:
    """The flow times tau^-1 maps `u` to, for a curve sampled on `ts`."""
    return make_tau_inverse(ts, recovery, vocab_size)(u)


def divisors_of(n: int, lo: int = 8) -> list[int]:
    """Candidate bin counts that subsample the reference grid exactly."""
    return [d for d in range(lo, n + 1) if n % d == 0]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """How far two sets of drawn t are apart, in t units.

    Summarised by quantiles of |dt| rather than its max. The map is steepest at the top
    knot, where tau^-1 climbs to t=1 over a sliver of u, so the single worst point moves by
    a lot for any change in the grid and says nothing about the other 99.9% of draws — it
    is not even monotone in the bin count.
    """
    diff = np.abs(a - b)
    return {
        "p50": float(np.percentile(diff, 50)),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
        **{f"q{q}": float(np.percentile(a, q) - np.percentile(b, q)) for q in QUANTILES},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Sweep size, reference resolution, and where to run it."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vocab-size", type=int, default=1193, help="1193 compact, 51866 full")
    p.add_argument("--n-utts", type=int, default=256)
    p.add_argument("--canvas-len", type=int, default=224)
    p.add_argument(
        "--reference", type=int, default=800,
        help="bins in the high-resolution sweep every candidate is subsampled from",
    )
    p.add_argument("--seeds", type=int, nargs=2, default=[0, 1], help="two, for the floor")
    p.add_argument(
        "--mu-accuracy", type=float, default=0.0,
        help="stand-in mu accuracy; with --gamma this is a refreshed simulated curve",
    )
    p.add_argument("--gamma", action="store_true", help="audio-conditioned path, gamma=1-t")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("figures/tau_bins.png"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep twice at high resolution, subsample, and report where the error flattens."""
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(0)
    tokens = torch.randint(
        0, args.vocab_size, (args.n_utts, args.canvas_len), device=device
    )
    ref_ts = np.linspace(0.0, 1.0, args.reference + 1)
    u = np.linspace(0.0, 1.0, 20001)

    print(f"vocabulary {args.vocab_size:,}  {args.n_utts} x {args.canvas_len} "
          f"= {args.n_utts * args.canvas_len:,} positions  on {device}")
    print(f"curve: {'audio-conditioned' if args.gamma else 'linear'}, "
          f"mu accuracy {args.mu_accuracy}")
    curves = {}
    for seed in args.seeds:
        started = time.perf_counter()
        curves[seed] = sweep(tokens, args.vocab_size, args.reference, seed,
                             mu_accuracy=args.mu_accuracy, gamma=args.gamma)
        print(f"  reference sweep seed {seed}: {args.reference + 1} points in "
              f"{time.perf_counter() - started:.1f}s")

    first, second = args.seeds
    ref = drawn_t(ref_ts, curves[first], args.vocab_size, u)
    floor = compare(drawn_t(ref_ts, curves[second], args.vocab_size, u), ref)
    print(f"\nnoise floor (two seeds, both at {args.reference} bins): "
          f"p50 {floor['p50']:.5f}  p95 {floor['p95']:.5f}  p99 {floor['p99']:.5f}")

    header = (f"{'bins':>6} {'p50':>9} {'p95':>9} {'p99':>9} " +
              " ".join(f"{'dq'+str(q):>7}" for q in QUANTILES) + "   verdict")
    print(f"\n{header}")
    rows, lines = [], []
    for n_bins in divisors_of(args.reference):
        step = args.reference // n_bins
        got = drawn_t(ref_ts[::step], curves[first][::step], args.vocab_size, u)
        r = compare(got, ref)
        rows.append((n_bins, r))
        verdict = "under the noise floor" if r["p95"] <= floor["p95"] else ""
        line = (f"{n_bins:>6} {r['p50']:9.5f} {r['p95']:9.5f} {r['p99']:9.5f} " +
                " ".join(f"{r['q'+str(q)]:+7.4f}" for q in QUANTILES) + f"   {verdict}")
        lines.append(line)
        print(line)

    enough = next((n for n, r in rows if r["p95"] <= floor["p95"]), None)
    answer = (f"smallest grid whose discretisation error is under the sampling noise: "
              f"{enough if enough else 'none tested'}")
    print(f"\n{answer}")

    # Next to the figure, because the figure alone cannot be read back as numbers and a
    # terminal scrollback is not somewhere results live.
    table = args.out.with_suffix(".txt")
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(
        f"vocabulary {args.vocab_size}  {args.n_utts} x {args.canvas_len} "
        f"= {args.n_utts * args.canvas_len} positions\n"
        f"reference {args.reference} bins, seeds {args.seeds}, shifts in units of t\n\n"
        f"noise floor: p50 {floor['p50']:.5f}  p95 {floor['p95']:.5f}  "
        f"p99 {floor['p99']:.5f}\n\n{header}\n" + "\n".join(lines) + f"\n\n{answer}\n"
    )
    print(f"wrote {table}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog([n for n, _ in rows], [r["p95"] for _, r in rows], "o-",
              color="navy", label="discretisation (p95 |dt| vs reference)")
    ax.loglog([n for n, _ in rows], [r["p50"] for _, r in rows], "s--",
              color="steelblue", ms=3, label="p50 |dt|")
    ax.axhline(floor["p95"], ls="--", color="crimson",
               label=f"sampling noise floor, p95 ({floor['p95']:.4f})")
    if enough:
        ax.axvline(enough, ls=":", color="grey", label=f"enough: {enough} bins")
    ax.set(xlabel="schedule.num_bins", ylabel="shift in drawn t",
           title=f"tau grid resolution, vocab {args.vocab_size:,}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
