"""How many metric bins the MEASURED tau refresh needs, and whether num_bins matters there.

`scripts/schedule/ablate_tau_bins.py` answers the question for the simulated curve, where the sweep
is evaluated wherever we ask and `schedule.num_bins` really is the resolution. The measured
refresh works nothing like that:

  * the curve comes from `BinnedTokenError`, so it has `metrics.perr_bins` points, not
    `schedule.num_bins` — ten, by default;
  * those points sit at schedule quantiles, so they bunch where the draw bunches, and each
    one is the mean t actually seen in its bin rather than a grid value;
  * each is an average over a finite number of tokens, so it carries real noise;
  * `schedule.num_bins` only sets the grid those points are INTERPOLATED ONTO, which cannot
    add information that was never measured.

So the binding resolution is `perr_bins`, and this measures it directly: take a truth
curve, bin it the way the metric would, add the sampling noise the metric would have, run
the real `MeasuredTau._tau_from`, and see how far the resulting tau^-1 moves.

Two truth shapes, because the answer depends on how curved the thing being sampled is:

  * `sweep`    — the real corruption sweep. Steep: flat, then a collapse around t~0.8.
  * `denoiser` — what a trained model looks like instead. Far shallower, because it reads
    the audio and is already ~0.25 wrong at low t where the interpolant is ~0.96 wrong.
    Synthetic, anchored on those two measured numbers; the shape between them is a guess,
    which is the one modelled assumption here.

    python -m scripts.schedule.ablate_measured_bins
    python -m scripts.schedule.ablate_measured_bins --truth denoiser --tokens-per-epoch 2000000
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.flow.schedule import make_tau_inverse, sweep_corruption  # noqa: E402
from src.flow.tau_measured import MeasuredTau  # noqa: E402

BIN_COUNTS = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150)
GRID_SIZES = (20, 50, 100, 400)


class FakeMetric:
    """A `BinnedTokenError` as `MeasuredTau._tau_from` sees it: centers, recovery, seen."""

    def __init__(self, centers: np.ndarray, recovery: np.ndarray) -> None:
        """Hold one refresh's worth of per-bin readings."""
        self._centers = torch.as_tensor(centers, dtype=torch.float32)
        self._recovery = torch.as_tensor(recovery, dtype=torch.float32)

    def seen(self) -> torch.Tensor:
        """Every bin populated; an empty one is a separate failure mode."""
        return torch.ones(self._centers.numel(), dtype=torch.bool)

    def bin_centers(self) -> torch.Tensor:
        """The mean t seen in each bin."""
        return self._centers

    def recovery(self) -> torch.Tensor:
        """Fraction of tokens recovered in each bin."""
        return self._recovery


def denoiser_truth(t: np.ndarray) -> np.ndarray:
    """A trained denoiser's recovery: ~0.75 at t=0, rising to ~1.

    The interpolant alone is 0.964 wrong in the lowest bin where a converged denoiser is
    0.251 wrong, because the denoiser hears the audio and the field does not. Those two
    endpoints are measured; the curve between them is a smooth guess.
    """
    return 0.749 + 0.251 * t**1.5


def binned_reading(
    truth_t: np.ndarray,
    truth_recovery: np.ndarray,
    tau_inverse: callable,
    n_bins: int,
    tokens_per_epoch: int,
    rng: np.random.Generator,
) -> FakeMetric:
    """Simulate one epoch of BinnedTokenError under the schedule's quantile bins.

    Edges are the schedule's quantiles, so each bin holds ~1/n of the epoch's tokens; the
    center is the mean t of the draws that landed in it, and the recovery is a binomial
    average over that bin's share of the tokens.
    """
    edges = tau_inverse(np.linspace(0.0, 1.0, n_bins + 1))
    drawn = tau_inverse(rng.random(200_000))
    idx = np.clip(np.searchsorted(edges[1:-1], drawn, side="right"), 0, n_bins - 1)

    centers, recoveries = [], []
    per_bin = max(tokens_per_epoch // n_bins, 1)
    for b in range(n_bins):
        in_bin = drawn[idx == b]
        center = float(in_bin.mean()) if in_bin.size else float((edges[b] + edges[b + 1]) / 2)
        p = float(np.interp(center, truth_t, truth_recovery))
        centers.append(center)
        recoveries.append(rng.binomial(per_bin, min(max(p, 0.0), 1.0)) / per_bin)
    return FakeMetric(np.array(centers), np.array(recoveries))


def drawn_from(metric: FakeMetric, vocab_size: int, num_bins: int, u: np.ndarray) -> np.ndarray:
    """Run the real MeasuredTau._tau_from and return the t it would draw."""
    tau = MeasuredTau(OmegaConf.create({"num_bins": num_bins}))
    return np.asarray(tau._tau_from(metric, vocab_size)(u))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which truth curve, how many tokens an epoch measures, and the vocabulary."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--truth", choices=["sweep", "denoiser"], default="denoiser")
    p.add_argument("--vocab-size", type=int, default=1193)
    p.add_argument(
        "--tokens-per-epoch", type=int, default=2_000_000,
        help="real tokens the binned metric averages over in one epoch, split across bins",
    )
    p.add_argument("--n-utts", type=int, default=128, help="only for --truth sweep")
    p.add_argument("--canvas-len", type=int, default=224)
    p.add_argument("--repeats", type=int, default=20, help="noise draws per bin count")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("figures/tau_measured_bins.png"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Bin a truth curve the way the metric would, and see what tau^-1 does."""
    args = parse_args(argv)
    truth_t = np.linspace(0.0, 1.0, 801)
    if args.truth == "denoiser":
        truth_recovery = denoiser_truth(truth_t)
    else:
        torch.manual_seed(0)
        tokens = torch.randint(
            0, args.vocab_size, (args.n_utts, args.canvas_len), device=args.device
        )
        truth_recovery = np.asarray(
            sweep_corruption(tokens, args.vocab_size, truth_t.tolist())
        )

    p_e0 = float(1.0 - truth_recovery[0])
    exact = make_tau_inverse(truth_t, truth_recovery, args.vocab_size, p_e0=p_e0)
    u = np.linspace(0.0, 1.0, 20001)
    ref = np.asarray(exact(u))

    print(f"truth: {args.truth}   vocabulary {args.vocab_size:,}   "
          f"{args.tokens_per_epoch:,} tokens/epoch   {args.repeats} noise draws")
    print(f"tau^-1 from the full {truth_t.size}-point curve is the reference\n")

    print(f"{'perr_bins':>10} {'p50 |dt|':>10} {'p95 |dt|':>10} {'spread':>9}   "
          f"(spread = p95 across noise draws)")
    rows = []
    for n_bins in BIN_COUNTS:
        p50s, p95s = [], []
        for rep in range(args.repeats):
            rng = np.random.default_rng(rep)
            metric = binned_reading(
                truth_t, truth_recovery, exact, n_bins, args.tokens_per_epoch, rng
            )
            got = drawn_from(metric, args.vocab_size, 100, u)
            d = np.abs(got - ref)
            p50s.append(float(np.percentile(d, 50)))
            p95s.append(float(np.percentile(d, 95)))
        rows.append((n_bins, float(np.median(p50s)), float(np.median(p95s)),
                     float(np.percentile(p95s, 95) - np.percentile(p95s, 5))))
        print(f"{n_bins:>10} {rows[-1][1]:10.4f} {rows[-1][2]:10.4f} {rows[-1][3]:9.4f}")

    print("\nsame reading, varying schedule.num_bins at perr_bins=10:")
    print(f"{'num_bins':>10} {'p50 |dt|':>10} {'p95 |dt|':>10}")
    grid_rows = []
    for grid in GRID_SIZES:
        rng = np.random.default_rng(0)
        metric = binned_reading(
            truth_t, truth_recovery, exact, 10, args.tokens_per_epoch, rng
        )
        d = np.abs(drawn_from(metric, args.vocab_size, grid, u) - ref)
        grid_rows.append((grid, float(np.percentile(d, 50)), float(np.percentile(d, 95))))
        print(f"{grid:>10} {grid_rows[-1][1]:10.4f} {grid_rows[-1][2]:10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.loglog([r[0] for r in rows], [r[2] for r in rows], "o-", color="navy", label="p95 |dt|")
    ax.loglog([r[0] for r in rows], [r[1] for r in rows], "s--", color="steelblue",
              ms=3, label="p50 |dt|")
    ax.axvline(10, ls=":", color="crimson", label="current perr_bins = 10")
    ax.set(xlabel="metrics.perr_bins", ylabel="shift in drawn t",
           title=f"measured refresh, truth = {args.truth}")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx([r[0] for r in grid_rows], [r[2] for r in grid_rows], "o-",
                color="navy", label="p95 |dt|")
    ax.axvline(100, ls=":", color="crimson", label="current num_bins = 100")
    ax.set(xlabel="schedule.num_bins (at perr_bins=10)", ylabel="shift in drawn t",
           title="the interpolation grid is not the constraint")
    ax.legend(fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
