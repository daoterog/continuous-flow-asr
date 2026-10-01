"""Check the claim the reparameterization is for: equal slices of u, equal decoding progress.

tau^-1 is built so that drawing u ~ U[0,1] and training at t = tau^-1(u) spreads the batch
evenly over *progress* rather than over time. If that works, the error rate at the drawn t
falls linearly in u, from P_e(0) down to 0 — a batch's worth of u covers the decoding
problem uniformly instead of piling up where nothing is decided.

The check has to be out of sample or it is circular: tau^-1 is fitted to one corruption
sweep, so measuring it against that same sweep would only confirm the interpolation. Here
the map is built from seed 0 and the error at each drawn t is re-measured under an
independent corruption, seed 1.

Uniform t is drawn alongside as the contrast — same measurement, no reparameterization.

    python -m scripts.schedule.verify_tau_linear
    python -m scripts.schedule.verify_tau_linear --vocab-size 51866 --n-utts 128
"""

import argparse
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.flow.schedule import make_tau_inverse, sweep_corruption  # noqa: E402


def measure_p_e(
    tokens: torch.Tensor, vocab_size: int, ts: np.ndarray, seed: int
) -> np.ndarray:
    """Error rate under the interpolant at each t, under a fresh corruption."""
    torch.manual_seed(seed)
    return 1.0 - np.asarray(sweep_corruption(tokens, vocab_size, ts.tolist()))


def straightness(u: np.ndarray, p_e: np.ndarray) -> dict:
    """How close P_e(u) is to the straight line the reparameterization promises.

    The ideal is P_e = P_e(0) * (1 - u): all of the error at u=0, none of it at u=1.
    Reported as R^2 against that *fixed* line, not against a free least-squares fit — a
    best-fit line would score well on any monotone curve, which is not the claim.
    """
    ideal = p_e[0] * (1.0 - u)
    resid = p_e - ideal
    ss_res = float((resid**2).sum())
    ss_tot = float(((p_e - p_e.mean()) ** 2).sum())
    return {
        "R2 vs ideal line": 1.0 - ss_res / max(ss_tot, 1e-12),
        "max |deviation|": float(np.abs(resid).max()),
        "mean |deviation|": float(np.abs(resid).mean()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Vocabulary, sweep size, and the resolution of the u grid."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vocab-size", type=int, default=1193, help="1193 compact, 51866 full")
    p.add_argument("--n-utts", type=int, default=256)
    p.add_argument("--canvas-len", type=int, default=224)
    p.add_argument("--num-bins", type=int, default=100, help="grid tau^-1 is built on")
    p.add_argument("--points", type=int, default=51, help="u values to measure at")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("figures/tau_linearity.png"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build tau^-1 on one sweep, then measure the error it actually lands on."""
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(0)
    tokens = torch.randint(
        0, args.vocab_size, (args.n_utts, args.canvas_len), device=device
    )

    fit_ts = np.linspace(0.0, 1.0, args.num_bins + 1)
    fit_recovery = 1.0 - measure_p_e(tokens, args.vocab_size, fit_ts, seed=0)
    tau_inverse = make_tau_inverse(fit_ts, fit_recovery, args.vocab_size)

    u = np.linspace(0.0, 1.0, args.points)
    t_tau = np.asarray(tau_inverse(u))
    # Both arms measured under the same fresh corruption, so the contrast is the schedule
    # and nothing else.
    p_e_tau = measure_p_e(tokens, args.vocab_size, t_tau, seed=1)
    p_e_uniform = measure_p_e(tokens, args.vocab_size, u, seed=1)

    print(f"vocabulary {args.vocab_size:,}  {args.n_utts} x {args.canvas_len} "
          f"= {args.n_utts * args.canvas_len:,} positions  on {device}")
    print(f"tau^-1 fitted on seed 0 ({args.num_bins} bins), "
          f"re-measured on seed 1 at {args.points} values of u\n")

    print(f"{'':<20} {'tau^-1':>12} {'uniform t':>12}")
    for name in ("R2 vs ideal line", "max |deviation|", "mean |deviation|"):
        a, b = straightness(u, p_e_tau), straightness(u, p_e_uniform)
        print(f"{name:<20} {a[name]:12.4f} {b[name]:12.4f}")

    print(f"\n{'u slice':<12} {'dP_e under tau^-1':>18} {'dP_e under uniform t':>22}")
    edges = np.linspace(0, args.points - 1, 6).astype(int)
    for lo, hi in pairwise(edges):
        print(f"{u[lo]:.1f} -> {u[hi]:.1f}   {p_e_tau[lo] - p_e_tau[hi]:18.4f} "
              f"{p_e_uniform[lo] - p_e_uniform[hi]:22.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.plot(u, p_e_tau, "o-", ms=3, color="seagreen", label=r"$t=\tau^{-1}(u)$")
    ax.plot(u, p_e_uniform, "s-", ms=3, color="grey", label="$t=u$ (uniform)")
    ax.plot(u, p_e_tau[0] * (1 - u), "--", color="crimson", lw=1,
            label=r"ideal: $P_e(0)(1-u)$")
    ax.set(xlabel="u (uniform draw)", ylabel="$P_e$ at the drawn t",
           title="error rate vs. draw")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(u, p_e_tau - p_e_tau[0] * (1 - u), "o-", ms=3, color="seagreen")
    ax.plot(u, p_e_uniform - p_e_uniform[0] * (1 - u), "s-", ms=3, color="grey")
    ax.axhline(0, ls="--", color="crimson", lw=1)
    ax.set(xlabel="u", ylabel="$P_e$ - ideal", title="deviation from linear")

    ax = axes[2]
    ax.plot(u, t_tau, color="seagreen")
    ax.plot(u, u, ls="--", color="grey", lw=1)
    ax.set(xlabel="u", ylabel="t", title=r"the map $\tau^{-1}$ being tested")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
