"""Figure: how mu's accuracy moves the interpolant's decoding error rate.

The audio-conditioned path is I_t = alpha(t) x0 + beta(t) x1 + gamma(t) mu. With
gamma = 1 - t, mu owns the low-t end, which is exactly where the noise drowns the target
and the field carries nothing on its own. This sweeps mu from useless to perfect and
measures what the interpolant's argmax recovers at each flow time — no model involved, so
the curves are a property of the probability path rather than of a trained denoiser.

The dashed baseline is the linear path (gamma = 0, no mu at all): the gap between it and
each blue curve is what conditioning buys at that accuracy.

    python -m scripts.schedule.plot_mu_sweep                       # sample of train.100
    python -m scripts.schedule.plot_mu_sweep --tau                 # add the tau panel
    python -m scripts.schedule.plot_mu_sweep --n-utts 512 --width 7.0 --out figures/mu.pdf

Writes a vector PDF for LaTeX plus a PNG to look at. Sized for a conference column by
default (3.4in), so type stays legible when the figure is not scaled.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.constants import data_root_for

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter
from transformers import WhisperTokenizerFast

from src.data import AudioFeatureDataModule
from src.flow.coefficients import make_schedule
from src.flow.schedule import (
    corrupt_and_decode,
    tau_from_recovery,
)

# Ordinal ramp, one hue light->dark, from the validated blue scale (steps 250/400/500/
# 600/700). mu accuracy is an ORDERED quantity, so it gets a sequential ramp rather than
# categorical hues — the reader should see "more mu" as "darker", not as "another series".
# Validated light-mode: monotone lightness, adjacent dL >= 0.06, light end 2.06:1 on the
# surface, hue spread 4 degrees.
MU_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"        # text-primary
INK_MUTED = "#52514e"  # text-secondary
GRID = "#e3e2de"
# Helvetica-like first (what most venues expect), then metric-compatible fallbacks.
FONT_STACK = ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """What to sweep and how big to draw it."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--config", default="clean", help="corpus config to draw tokens from")
    p.add_argument("--split", default="train.100", help="corpus split to draw tokens from")
    p.add_argument("--n-utts", type=int, default=256, help="utterances in the sweep")
    p.add_argument("--canvas-len", type=int, default=224)
    p.add_argument(
        "--accuracies", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="mu accuracies to simulate; at most as many as the ramp has steps",
    )
    p.add_argument("--num-t", type=int, default=81, help="points along the t axis")
    p.add_argument("--tau", action="store_true", help="add the tau(t) panel beside it")
    p.add_argument("--width", type=float, default=3.4, help="figure width in inches")
    p.add_argument("--out", type=Path, default=Path("figures/mu_sweep.pdf"))
    p.add_argument(
        "--font", default=None,
        help="sans family for text AND math; defaults to the first of "
             f"{FONT_STACK} that is installed",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def argmax_threshold(vocab_size: int) -> float:
    """The t above which the interpolant's argmax starts recovering the true token.

    The crossover sits where the signal beats the largest of V draws from N(0, 1-t):
    t/(1-t) > E[max], i.e. t > E[max]/(1 + E[max]). sqrt(2 ln V) is the leading-order
    term and runs a little high — measured, E[max] is 4.24 at V=51,866 against a
    predicted 4.66.
    """
    e_max = np.sqrt(2 * np.log(vocab_size))
    return float(e_max / (1 + e_max))


def load_tokens(args: argparse.Namespace) -> tuple["object", int]:
    """(n_utts, canvas_len) targets and the compact vocabulary size."""
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-large-v3")
    role = {"configs": [args.config], "splits": [args.split]}
    dm = AudioFeatureDataModule(
        data_root=data_root_for(args.env),
        model="openai/whisper-large-v3",
        splits={"train": role, "val": role, "test": role},
        tokenizer=tokenizer,
        canvas_len=args.canvas_len,
        batch_size=8,
        subset_n={"train": args.n_utts, "val": 8, "test": 8},
    ).prepare()
    return dm.sample_tokens("train", args.n_utts, seed=args.seed), dm.vocab_size


def error_curve(
    tokens: "object",
    vocab_size: int,
    ts: "object",
    alpha: "object",
    beta: "object",
    gamma: "object",
    accuracy: float,
    seed: int,
) -> "object":
    """P_e(t) along `ts`, with one noise realisation shared by every point.

    src.flow.schedule.sweep_corruption draws fresh noise per t, which is right for measuring
    tau — each point is then an independent estimate. For a figure it is wrong: the
    per-point sampling error shows up as jitter along a curve that is smooth in truth,
    and at 41 points it reads as if the quantity itself were noisy.

    Reseeding to the same value before each t makes this common random numbers: the field
    is the identical draw with only t moved, so the curve is smooth and the accuracies are
    directly comparable rather than each carrying its own noise. It does not change what
    is being estimated, only the variance between neighbouring points.
    """
    out = []
    for t in ts:
        generator = torch.Generator().manual_seed(seed)
        pred = corrupt_and_decode(
            tokens, vocab_size, float(t), alpha, beta, gamma, accuracy,
            generator=generator,
        )
        out.append((pred != tokens).float().mean().item())
    return np.asarray(out)


def pick_font(preferred: str | None = None) -> str:
    """First available sans family, so the figure looks the same on laptop and cluster."""
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ([preferred] if preferred else []) + FONT_STACK:
        if name in have:
            return name
    return "DejaVu Sans"   # ships with matplotlib, so it is always there


def style(font: str) -> None:
    """Print-first defaults: no chartjunk, type large enough to survive a column.

    mathtext is pinned to the SAME family as the body text. Matplotlib renders math with
    its own font set regardless of font.sans-serif, so the default mixes two typefaces
    inside a single label — "decoding error" in one and "$P_e(t)$" in another. pdffonts
    shows it as three embedded fonts where there should be one.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, *FONT_STACK],
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,      # embed as TrueType, not Type 3 — required by most venues
        "ps.fonttype": 42,
    })


def draw_axes(ax: "plt.Axes") -> None:
    """Recessive frame: horizontal rules only, no top or right spine."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweeps and write the figure."""
    args = parse_args(argv)
    if len(args.accuracies) > len(MU_RAMP):
        print(f"at most {len(MU_RAMP)} accuracies (the ramp has that many validated steps)")
        return 1

    tokens, vocab_size = load_tokens(args)
    ts = np.linspace(0.0, 1.0, args.num_t)
    alpha, beta = make_schedule("linear_down"), make_schedule("linear_up")
    gamma = make_schedule("linear_down")  # audio_cond: gamma = 1 - t

    print(f"sweeping {tokens.shape[0]} utts, V={vocab_size}, {args.num_t} points of t")
    curves = {}
    for acc in args.accuracies:
        curves[acc] = error_curve(
            tokens, vocab_size, ts, alpha, beta, gamma, acc, args.seed
        )
        print(f"  mu accuracy {acc:>4.0%}: P_e(0)={curves[acc][0]:.3f} "
              f"P_e(0.5)={curves[acc][len(ts)//2]:.3f}")
    # The linear path is measured but not drawn. It is NOT the same construct as an
    # accuracy-0 mu: simulate_mu(0.0) emits a confident one-hot on a uniformly random
    # token, so the field gets a wrong answer stated at full confidence, where the linear
    # path gets no third term at all. They measure the same because that wrong spike is
    # only gamma(t) ~ 1 tall and the noise maximum alpha*sqrt(2 ln V) is ~3.8, so it is
    # drowned rather than absent. Drawing both would put two identical lines on the plot
    # and read as a rendering fault, so the agreement is printed instead.
    linear = error_curve(tokens, vocab_size, ts, alpha, beta, None, 1.0, args.seed)
    print(f"  linear (no mu):  P_e(0)={linear[0]:.3f} P_e(0.5)={linear[len(ts)//2]:.3f}"
          f"   <- not drawn; it coincides with mu accuracy 0.00")

    font = pick_font(args.font)
    style(font)
    print(f"  font: {font} (text and math)")
    ncols = 2 if args.tau else 1
    # --width is the width of one panel, so a two-panel figure is twice as wide rather
    # than two squeezed halves: the point of the second panel is that it stays readable.
    fig, axes = plt.subplots(
        1, ncols, figsize=(args.width * ncols, args.width * 0.70), squeeze=False
    )
    ax = axes[0][0]
    draw_axes(ax)

    # Where the target's spike starts beating the loudest of V noise draws. Left of it the
    # linear path carries nothing, which is the region the third term exists to fill — so
    # it is shaded rather than ruled: it is an interval, not a boundary.
    thr = argmax_threshold(vocab_size)
    ax.axvspan(0, thr, color=GRID, alpha=0.35, linewidth=0, zorder=0)
    ax.axvline(thr, color=INK_MUTED, linewidth=0.5, linestyle=(0, (1, 2.5)), zorder=1)
    # Placed in axes fractions, in the wedge between the legend and the curves: the
    # curves hug the top across the whole shaded half, so there is no room above them.
    ax.annotate("dead zone without $\\mu$", xy=(0.035, 0.58), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=6.5, color=INK_MUTED)

    for acc, color in zip(args.accuracies, MU_RAMP, strict=False):
        ax.plot(ts, curves[acc], color=color, linewidth=1.6, solid_capstyle="round",
                zorder=3, label=f"{acc:.0%}")

    ax.set_xlabel("flow time $t$")
    ax.set_ylabel("decoding error $P_e(t)$")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    leg = ax.legend(title="$\\mu$ accuracy", loc="lower left", frameon=False,
                    handlelength=1.1, labelspacing=0.22, borderpad=0.15,
                    handletextpad=0.45, reverse=True)
    leg.get_title().set_fontsize(6.8)
    leg.get_title().set_color(INK_MUTED)
    for text in leg.get_texts():
        text.set_color(INK_MUTED)

    if args.tau:
        ax2 = axes[0][1]
        draw_axes(ax2)
        ax2.axvspan(0, thr, color=GRID, alpha=0.35, linewidth=0, zorder=0)
        ax2.axvline(thr, color=INK_MUTED, linewidth=0.5, linestyle=(0, (1, 2.5)), zorder=1)
        for acc, color in zip(args.accuracies, MU_RAMP, strict=False):
            tau = tau_from_recovery(1.0 - curves[acc], vocab_size)
            ax2.plot(ts, tau, color=color, linewidth=1.6, solid_capstyle="round", zorder=3)
        ax2.set_xlabel("flow time $t$")
        ax2.set_ylabel(r"decoded fraction $\tau(t)$")
        ax2.set_ylim(-0.02, 1.02)
        # Same units as the left panel, so the reader is not switching scales mid-figure.
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    fig.tight_layout(pad=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    png = args.out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    print(f"\nwrote {args.out}  and  {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
