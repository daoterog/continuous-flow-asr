"""Check the closed-form recovery curve against the Monte-Carlo sweep it replaces.

`src/flow/recovery.py` integrates what `schedule.sweep_corruption` samples. If the
derivation is right the two agree to the sweep's own noise, so the test is not "are they
close" but "is the disagreement smaller than the sweep could resolve anyway".

The sweep's standard error on a recovery probability is ~1/sqrt(N) over N canvas
positions, and tau divides by P_e(0) ~ 1, so that carries straight through to tau. Every
disagreement below is reported as a multiple of that sigma; anything around 1 is the sweep
being noisy, not the integral being wrong.

Checked for all three interpolant variants, because the mu term is where the derivation
stops being FMLM's: with mu one-hot, recovery conditions on whether it is right, and the
wrong branch leaves one rival slot raised by gamma instead of V-1 rivals at zero.

    python -m scripts.schedule.verify_recovery
    python -m scripts.schedule.verify_recovery --vocab-size 51866 --n-utts 512
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.flow.coefficients import make_schedule
from src.flow.recovery import recovery_curve
from src.flow.schedule import sweep_corruption, tau_from_recovery

# The variants a run can actually be configured with, at mu accuracies spanning useless to
# perfect. gamma is given as the config spec so these are the same callables training uses.
VARIANTS: list[tuple[str, dict | str | None, float]] = [
    ("linear", None, 0.0),
    ("audio_cond a=0.0", "linear_down", 0.0),
    ("audio_cond a=0.5", "linear_down", 0.5),
    ("audio_cond a=0.9", "linear_down", 0.9),
    ("audio_cond a=1.0", "linear_down", 1.0),
    ("three_term c=0.5 a=0.0", {"type": "parabola", "c": 0.5}, 0.0),
    ("three_term c=0.5 a=0.5", {"type": "parabola", "c": 0.5}, 0.5),
    ("three_term c=0.5 a=0.9", {"type": "parabola", "c": 0.5}, 0.9),
    ("three_term c=0.25 a=0.7", {"type": "parabola", "c": 0.25}, 0.7),
]


def tau_pair(
    tokens: torch.Tensor,
    vocab_size: int,
    ts: np.ndarray,
    alpha_fn: Callable,
    beta_fn: Callable,
    gamma_fn: Callable | None,
    mu_accuracy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(exact, swept) tau on the same grid, so only the method differs."""
    exact = recovery_curve(ts, vocab_size, alpha_fn, beta_fn, gamma_fn, mu_accuracy)
    swept = sweep_corruption(
        tokens, vocab_size, ts.tolist(), alpha_fn, beta_fn, gamma_fn, mu_accuracy
    )
    return (
        tau_from_recovery(exact, vocab_size),
        tau_from_recovery(swept, vocab_size),
    )


# numpy's hermgauss builds its weights as 1/(fm*fm), which overflows past ~300 nodes and
# returns NaN with only a RuntimeWarning. 300 is the most it survives, so it is the
# reference; N_NODES has to stay comfortably under it.
REFERENCE_NODES = 300


def convergence(
    ts: np.ndarray, vocab_size: int, alpha_fn: Callable, beta_fn: Callable
) -> list[tuple[int, float]]:
    """How many quadrature nodes the curve actually needs, against a 300-node reference.

    Run on the mu branch, which has the two-shift integrand and so is the harder of the
    two to resolve.
    """
    gamma_fn = make_schedule("linear_down")
    ref = recovery_curve(
        ts, vocab_size, alpha_fn, beta_fn, gamma_fn, 0.5, n_gh=REFERENCE_NODES
    )
    if not np.all(np.isfinite(ref)):
        raise RuntimeError(
            f"hermgauss({REFERENCE_NODES}) returned non-finite weights; lower "
            f"REFERENCE_NODES"
        )
    out = []
    for n in (10, 20, 40, 60, 100, 200):
        got = recovery_curve(ts, vocab_size, alpha_fn, beta_fn, gamma_fn, 0.5, n_gh=n)
        out.append((n, float(np.abs(got - ref).max())))
    return out


def shape_checks(
    vocab_size: int, alpha_fn: Callable, beta_fn: Callable, points: int = 10001
) -> list[tuple[str, str]]:
    """Endpoints, and where in t the curve stops being strictly increasing.

    The interesting question is not whether float64 holds monotonicity — it does not, at
    the top — but WHERE it breaks, because that tells the two halves of the tau machinery
    apart. Measured on the raw curve, before `tau_from_recovery` clips and accumulates:

      * a break only above some t_sat means recovery has saturated to 1.0 and tau to 1,
        which is a fact about the path and is what `generation_grid`'s collapse of
        repeated t handles;
      * a break at the BOTTOM would be the thing `flat_eps` exists for. The exact curve
        has none, which is why `tau^-1(0)` can be taken at face value.

    `starts above chance` is the `p_e0` normaliser: recovery(0) > 1/V means gamma(0) != 0,
    the path begins with mu already in the field, and `1 - 1/V` is the wrong denominator.
    """
    ts = np.linspace(0.0, 1.0, points)
    rows = []
    for name, spec, acc in (
        ("linear", None, 0.0),
        ("audio_cond", "linear_down", 0.9),
        ("three_term", {"type": "parabola", "c": 0.5}, 0.9),
    ):
        r = recovery_curve(ts, vocab_size, alpha_fn, beta_fn, make_schedule(spec), acc)
        raw = 1.0 - (1.0 - r) / (1.0 - 1.0 / vocab_size)
        rising = np.diff(raw) > 0
        first_flat = ts[int(np.argmin(rising))] if not rising.all() else None
        chance = 1.0 / vocab_size
        rows.append(
            (
                name,
                f"r(0)={r[0]:.6f} (1/V={chance:.6f}, "
                f"{'ABOVE chance' if r[0] > chance * 1.01 else 'at chance'})  "
                f"r(1)={r[-1]:.8f}  tau(0)={raw[0]:.6f}\n"
                f"  {'':<12} increasing up to "
                f"{'t=1.0 (everywhere)' if first_flat is None else f't={first_flat:.4f}'}"
                f", flat for {(~rising).sum()}/{len(rising)} steps above it"
                f"  finite={bool(np.all(np.isfinite(r)))}",
            )
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Vocabulary, how big a sweep to contest the integral with, and where to plot it."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vocab-size", type=int, default=1193, help="1193 compact, 51866 full")
    p.add_argument("--n-utts", type=int, default=256)
    p.add_argument("--canvas-len", type=int, default=224)
    p.add_argument("--points", type=int, default=51, help="values of t to compare at")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("figures/recovery_exact.png"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the two methods on every variant, then plot the three representative ones."""
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(0)
    tokens = torch.randint(
        0, args.vocab_size, (args.n_utts, args.canvas_len), device=device
    )

    alpha_fn = make_schedule("linear_down")
    beta_fn = make_schedule("linear_up")
    ts = np.linspace(0.0, 1.0, args.points)

    n_pos = args.n_utts * args.canvas_len
    sigma = 1.0 / np.sqrt(n_pos)
    print(f"vocabulary {args.vocab_size:,}  {args.n_utts} x {args.canvas_len} "
          f"= {n_pos:,} positions per sweep point  on {device}")
    print(f"sweep's 1-sigma on tau ~ {sigma:.5f}; disagreements are shown against it\n")

    print(f"{'variant':<26} {'max |dtau|':>11} {'mean |dtau|':>12} {'max/sigma':>10}")
    curves = {}
    for name, spec, acc in VARIANTS:
        gamma_fn = make_schedule(spec)
        exact, swept = tau_pair(
            tokens, args.vocab_size, ts, alpha_fn, beta_fn, gamma_fn, acc
        )
        curves[name] = (exact, swept)
        d = np.abs(exact - swept)
        print(f"{name:<26} {d.max():>11.5f} {d.mean():>12.5f} {d.max() / sigma:>10.2f}")

    print(f"\nquadrature convergence (mu branch, vs {REFERENCE_NODES} nodes):")
    for n, err in convergence(ts, args.vocab_size, alpha_fn, beta_fn):
        print(f"  n_gh={n:<4} max |diff| = {err:.3e}")

    print("\nshape, on a 10,001-point grid:")
    for name, line in shape_checks(args.vocab_size, alpha_fn, beta_fn):
        print(f"  {name:<12} {line}")

    shown = ["linear", "audio_cond a=0.9", "three_term c=0.5 a=0.9"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
    for col, name in enumerate(shown):
        exact, swept = curves[name]
        ax = axes[0, col]
        ax.plot(ts, swept, "s", ms=3, color="grey", label="sweep (Monte Carlo)")
        ax.plot(ts, exact, "-", color="seagreen", lw=1.6, label="closed form")
        ax.set(ylabel=r"$\tau$" if col == 0 else None, title=name)
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(ts, exact - swept, "o-", ms=2.5, color="seagreen")
        ax.axhspan(-sigma, sigma, color="crimson", alpha=0.15,
                   label=r"sweep $\pm 1\sigma$" if col == 0 else None)
        ax.axhline(0, ls="--", color="crimson", lw=1)
        ax.set(xlabel="t", ylabel="exact - swept" if col == 0 else None)
        if col == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
