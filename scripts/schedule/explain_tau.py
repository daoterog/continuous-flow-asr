"""Rebuild tau^-1 the obvious way, plot every stage, and check it against the real one.

`src.flow.schedule.make_tau_inverse` is written for the training loop: it has to run on
GPU tensors every step without syncing, so the interpolation is hand-rolled. This does the
same thing with `np.interp` and no device handling, which is readable, and then asserts the
two agree so the fast one needs no further trust.

The four panels, left to right, are the pipeline:

    recovery(t) -> P_e(t) -> tau(t) -> tau^-1(u) -> the t that training draws

    python -m scripts.schedule.explain_tau
    python -m scripts.schedule.explain_tau --vocab-size 51866 --n-utts 128
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.flow.schedule import make_tau_inverse, sweep_corruption

FLAT_EPS = 1e-3


def readable_tau(
    recovery: np.ndarray, vocab_size: int, p_e0: float | None = None
) -> np.ndarray:
    """tau(t): the fraction of the decoding that has happened by flow time t.

    Three steps, each one line:

    1. P_e(t) = 1 - recovery(t). The per-token error rate at that corruption level.
    2. tau = 1 - P_e(t) / P_e(0). At t=0 nothing is decoded, so P_e(t) = P_e(0) and tau=0;
       once the field is clean P_e -> 0 and tau -> 1. So tau reads as progress, not time.
    3. force it non-decreasing. Progress cannot go backwards, but P_e is estimated from a
       finite sample, so it wobbles; `maximum.accumulate` keeps the running best.

    `p_e0` is what counts as "no progress". Chance (1 - 1/V) for the corruption sweep,
    where a model-free argmax really is guessing at t=0. For the denoiser's own curve it is
    that curve's worst error instead, because the denoiser hears the audio and is nowhere
    near chance at t=0 — normalising by chance would invent a dead zone it does not have.
    """
    p_e = 1.0 - np.asarray(recovery, dtype=np.float64)
    if p_e0 is None:
        p_e0 = 1.0 - 1.0 / vocab_size
    tau = np.clip(1.0 - p_e / max(p_e0, 1e-12), 0.0, 1.0)
    return np.maximum.accumulate(tau)


def readable_knots(
    ts: np.ndarray, tau: np.ndarray, flat_eps: float = FLAT_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """The (tau, t) pairs to interpolate through, with the flat dead zone collapsed.

    Inverting a function means reading its graph with the axes swapped: tau(t) goes t ->
    tau, so interpolating with tau as x and t as y gives t = tau^-1(u). The catch is that
    interpolation needs x to be strictly increasing, and tau is pinned at 0 across the
    whole dead zone — at Whisper's vocabulary that is every t below ~0.65.

    So two cleanups:

    * snap near-zero tau to exactly zero. Sampling noise leaves ~1e-4 wiggles down there,
      and step 3 above has already frozen them in as real-looking increases. Left alone
      they become genuine knots, and training draws land inside the dead zone.
    * of each run of equal tau, keep only the LAST t. That is what makes tau^-1(0) the
      right edge of the dead zone rather than its left edge. Keeping the first instead
      would map the smallest draws to t=0 and spread mass across a region where the noise
      drowns the target and nothing is decided.
    """
    tau = tau.copy()
    tau[tau < flat_eps] = 0.0
    rises_next = np.concatenate([np.diff(tau) > 0, [True]])
    return tau[rises_next], ts[rises_next]


def readable_tau_inverse(ts: np.ndarray, tau: np.ndarray) -> "callable":
    """Map a uniform draw u to a flow time t, by interpolating through the knots."""
    tau_k, t_k = readable_knots(ts, tau)

    def tau_inverse(u: np.ndarray) -> np.ndarray:
        # np.interp clamps outside the knots: u below tau_k[0] gives t_k[0], above the
        # last gives t_k[-1]. That clamp is what keeps draws out of the dead zone.
        return np.interp(np.asarray(u, dtype=np.float64), tau_k, t_k)

    return tau_inverse


def compare(ts: np.ndarray, recovery: np.ndarray, vocab_size: int) -> dict:
    """Largest disagreement between this file's tau^-1 and the one training uses."""
    mine = readable_tau_inverse(ts, readable_tau(recovery, vocab_size))
    theirs = make_tau_inverse(ts, recovery, vocab_size)

    u = np.linspace(0.0, 1.0, 20001)
    got_np, want = theirs(u), mine(u)
    got_torch = theirs(torch.as_tensor(u)).numpy()

    rng = np.random.default_rng(0)
    u_rand = rng.random(100_000)
    # make_tau_inverse stores tau already clamped, so compare against the clamped form;
    # the gap before clamping is the dead-zone noise and is bounded by flat_eps.
    tau_raw = readable_tau(recovery, vocab_size)
    tau_clamped = tau_raw.copy()
    tau_clamped[tau_clamped < FLAT_EPS] = 0.0
    return {
        "tau (clamped)": float(np.abs(tau_clamped - theirs.tau).max()),
        "noise flat_eps removed": float(np.abs(tau_raw - tau_clamped).max()),
        "numpy path": float(np.abs(got_np - want).max()),
        "torch path": float(np.abs(got_torch - want).max()),
        "random draws": float(np.abs(theirs(u_rand) - mine(u_rand)).max()),
        "knots kept": theirs.knots[0].size,
        "grid points": ts.size,
    }


def plot(ts: np.ndarray, recovery: np.ndarray, vocab_size: int, out: Path) -> None:
    """The pipeline in four panels, so each stage can be eyeballed."""
    tau = readable_tau(recovery, vocab_size)
    tau_k, t_k = readable_knots(ts, tau)
    tau_inverse = readable_tau_inverse(ts, tau)
    first_knot = float(t_k[0])

    rng = np.random.default_rng(0)
    drawn = tau_inverse(rng.random(200_000))
    q01, q50 = np.percentile(drawn, [1, 50])

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))

    ax = axes[0]
    ax.plot(ts, 1.0 - recovery, color="crimson")
    ax.set(xlabel="t", ylabel="$P_e(t)$", title="1. error rate under the interpolant")

    ax = axes[1]
    ax.plot(ts, tau, color="navy")
    ax.axhline(FLAT_EPS, ls="--", color="crimson", lw=0.8)
    ax.axvline(first_knot, ls=":", color="grey")
    ax.set(xlabel="t", xscale="linear", yscale="log", ylim=(1e-5, 1.5),
           ylabel=r"$\tau(t)$  (log)", title="2. decoding progress")
    ax.annotate(
        f"first knot: $\\tau$ crosses\nflat_eps at t={first_knot:.2f}",
        xy=(first_knot, FLAT_EPS), xytext=(0.3, 2e-5), fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "grey"},
    )

    ax = axes[2]
    u = np.linspace(0, 1, 2000)
    ax.plot(u, tau_inverse(u), color="seagreen")
    ax.plot(tau_k, t_k, "o", ms=2.5, color="black", label=f"{tau_k.size} knots")
    ax.set(xlabel="u (uniform draw)", ylabel="t", ylim=(0, 1),
           title=r"3. $\tau^{-1}$: swap the axes")
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.hist(drawn, bins=80, range=(0, 1), color="seagreen", alpha=0.8, label=r"$\tau^{-1}(u)$")
    ax.hist(rng.random(200_000), bins=80, range=(0, 1), histtype="step",
            color="grey", label="uniform t")
    ax.axvline(q01, ls=":", color="black", lw=0.9)
    ax.axvline(q50, ls="--", color="black", lw=0.9)
    ax.set(xlabel="t drawn for training", ylabel="count", title="4. where training lands")
    ax.annotate(f"1% below\nt={q01:.2f}", xy=(q01, 0), xytext=(0.03, 0.72),
                textcoords="axes fraction", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "grey"})
    ax.annotate(f"median\nt={q50:.2f}", xy=(q50, 0), xytext=(0.62, 0.72),
                textcoords="axes fraction", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Vocabulary and sweep size for the demonstration curve."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vocab-size", type=int, default=1193, help="1193 compact, 51866 full")
    p.add_argument("--n-utts", type=int, default=128)
    p.add_argument("--canvas-len", type=int, default=224, help="the real canvas")
    p.add_argument("--num-bins", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("figures/tau_explained.png"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep a corruption curve, rebuild tau^-1 from it, plot it, and check it."""
    args = parse_args(argv)
    torch.manual_seed(0)
    tokens = torch.randint(0, args.vocab_size, (args.n_utts, args.canvas_len))
    ts = np.linspace(0.0, 1.0, args.num_bins + 1)
    recovery = np.asarray(sweep_corruption(tokens, args.vocab_size, ts.tolist()))

    print(f"vocabulary {args.vocab_size:,}  {args.n_utts} utterances x {args.canvas_len}\n")
    plot(ts, recovery, args.vocab_size, args.out)

    tau_inv = make_tau_inverse(ts, recovery, args.vocab_size)
    rng = np.random.default_rng(0)
    drawn = tau_inv(rng.random(200_000))
    qs = np.percentile(drawn, [1, 10, 50, 90, 99])
    print("t that training draws:  " + "  ".join(
        f"q{q}={v:.3f}" for q, v in zip([1, 10, 50, 90, 99], qs, strict=True)))

    print("\nreadable vs. make_tau_inverse, largest absolute difference:")
    for name, value in compare(ts, recovery, args.vocab_size).items():
        print(f"  {name:<14} {value:.3e}" if isinstance(value, float) else
              f"  {name:<14} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
