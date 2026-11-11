"""Figure: how mu's accuracy and the noise scale sigma move the interpolant's decoding error.

The audio-conditioned path is I_t = alpha(t) x0 + beta(t) x1 + gamma(t) mu, with
alpha = sigma (1 - t) and gamma = 1 - t. mu owns the low-t end, where the noise drowns the
target and the field carries nothing on its own; how much of mu shows through depends on
both how often mu is right and how loud the noise is. Two panels, one knob each:

  left    sigma fixed (the recipe's 0.4), mu's accuracy varied
  right   mu's accuracy fixed (0.45, what mu reaches in our runs), sigma varied

Each curve is what the interpolant's argmax recovers at each flow time — no model involved,
so the curves are a property of the probability path rather than of a trained denoiser.
mu is simulated as a one-hot that is right with the given probability, the same
approximation the simulated tau makes. The dashed baseline on both panels is the linear
path as the linear recipe trains it (gamma = 0, no mu, unit noise).

    python -m scripts.schedule.plot_mu_sweep                       # sample of train.100
    python -m scripts.schedule.plot_mu_sweep --sigmas 1 0.5 0.4 0.2 0.05 --mu-accuracy 0.45
    python -m scripts.schedule.plot_mu_sweep --n-utts 512 --width 3.4 --out figures/mu.pdf

The curves are cached next to the figure (`<out>.npz`) with the settings behind them, so
restyling redraws in seconds; a changed setting, or --resimulate, reruns the sweep.

Writes a vector PDF for LaTeX plus a PNG to look at. --width is one panel's width, sized
for a conference column by default (3.4in), so type stays legible when not scaled.
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
from matplotlib import patheffects
from matplotlib.ticker import PercentFormatter
from transformers import WhisperTokenizerFast

from src.data import AudioFeatureDataModule
from src.flow.coefficients import make_schedule
from src.flow.schedule import corrupt_and_decode

# Ordinal ramp, one hue light->dark, from the validated blue scale (steps 250/400/500/
# 600/700). Both knobs are ORDERED, so each gets a sequential ramp rather than categorical
# hues, oriented the same way on both panels: darker means more of mu shows through —
# a more accurate mu on the left, less noise on the right.
# Validated light-mode: monotone lightness, adjacent dL >= 0.06, light end 2.06:1 on the
# surface, hue spread 4 degrees.
# Palette shared with figure_method.tex: mu blue (2A5BD7), mulight (9DB6F2), ink, enc, panel.
# Ordinal ramp, light -> dark: 0% is mulight, 50% is exactly mu, then two darker steps.
# Sans families tried in order for text and math alike; DejaVu ships with matplotlib.
FONT_STACK = ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
MU_RAMP = ["#9DB6F2", "#6489E4", "#2A5BD7", "#1E409C", "#132766"]
INK = "#1B1F2A"  # ink
INK_MUTED = "#5F6B85"  # enc
GRID = "#DDE3EE"  # horizontal rules (lighter than enc)
# The recipe's own setting, wherever it appears among the varied curves: off the blue ramp
# entirely, so the configuration actually trained is unmistakable.
CHOSEN = "#5B2DB3"


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
        "--accuracies",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="left panel: mu accuracies to simulate, at most as many as the ramp has steps",
    )
    p.add_argument("--num-t", type=int, default=81, help="points along the t axis")
    p.add_argument(
        "--sigma",
        type=float,
        default=0.4,
        help="left panel: the fixed noise scale, alpha = sigma (1 - t); the recipe's 0.4",
    )
    p.add_argument(
        "--mu-accuracy",
        type=float,
        default=0.45,
        help="right panel: the fixed mu accuracy; mu reaches ~0.45 in our runs",
    )
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[1.0, 0.7, 0.4, 0.2, 0.05],
        help="right panel: noise scales to simulate, at most as many as the ramp has steps",
    )
    p.add_argument("--width", type=float, default=3.4, help="one panel's width in inches")
    p.add_argument("--out", type=Path, default=Path("figures/mu_sweep.pdf"))
    p.add_argument(
        "--font",
        default=None,
        help="sans family for text AND math; defaults to the first of "
        f"{FONT_STACK} that is installed",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resimulate",
        action="store_true",
        help="rerun the sweep even when a cache next to --out matches these settings",
    )
    return p.parse_args(argv)


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
            tokens,
            vocab_size,
            float(t),
            alpha,
            beta,
            gamma,
            accuracy,
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
    return "DejaVu Sans"  # ships with matplotlib, so it is always there


def style(font: str) -> None:
    """Print-first defaults: no chartjunk, type large enough to survive a column.

    mathtext is pinned to the SAME family as the body text. Matplotlib renders math with
    its own font set regardless of font.sans-serif, so the default mixes two typefaces
    inside a single label — "decoding error" in one and "$P_e(t)$" in another. pdffonts
    shows it as three embedded fonts where there should be one.
    """
    plt.rcParams.update(
        {
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
            "pdf.fonttype": 42,  # embed as TrueType, not Type 3 — required by most venues
            "ps.fonttype": 42,
        }
    )


def draw_axes(ax: "plt.Axes") -> None:
    """Recessive frame: horizontal rules only, no top or right spine."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])


def draw_curves(
    ax: "plt.Axes",
    ts: "object",
    linear: "object",
    curves: dict,
    label: "object",
    title: str,
    legend_title: str,
    chosen: float | None = None,
) -> None:
    """One panel: the dashed no-mu baseline under one ramp of curves, darkest drawn last.

    The curve whose key equals `chosen` — the recipe's setting — is drawn in CHOSEN, on
    top, and marked in the legend.
    """
    draw_axes(ax)
    ax.plot(
        ts,
        linear,
        color=INK_MUTED,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        zorder=2,
        label="no $\\mu$, $\\sigma=1$",
    )
    for (key, curve), color in zip(curves.items(), MU_RAMP, strict=False):
        is_chosen = chosen is not None and np.isclose(key, chosen)
        ax.plot(
            ts,
            curve,
            color=CHOSEN if is_chosen else color,
            linewidth=2.0 if is_chosen else 1.6,
            solid_capstyle="round",
            zorder=4 if is_chosen else 3,
            label=f"{label(key)} (ours)" if is_chosen else label(key),
        )
    # A medium weight, between the font's only two (400 and 700): the regular face
    # thickened by a hairline stroke in its own colour. Bold read as too heavy.
    ax.set_title(
        title,
        loc="center",
        color="black",
        pad=10,
        path_effects=[patheffects.withStroke(linewidth=0.15, foreground="black")],
    )
    ax.set_xlabel("flow time $t$")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    leg = ax.legend(
        title=legend_title,
        loc="lower left",
        frameon=False,
        handlelength=1.1,
        labelspacing=0.22,
        borderpad=0.15,
        handletextpad=0.45,
        reverse=True,
    )
    leg.get_title().set_fontsize(6.8)
    leg.get_title().set_color(INK_MUTED)
    for text in leg.get_texts():
        text.set_color(INK_MUTED)


def settings(args: argparse.Namespace) -> dict:
    """Everything the curves depend on, stored with them so a stale cache is never drawn."""
    return {
        "corpus": np.asarray(f"{args.config}/{args.split}"),
        "n_utts": np.asarray(args.n_utts),
        "canvas_len": np.asarray(args.canvas_len),
        "num_t": np.asarray(args.num_t),
        "seed": np.asarray(args.seed),
        "sigma": np.asarray(args.sigma),
        "mu_accuracy": np.asarray(args.mu_accuracy),
        "accuracies": np.asarray(sorted(args.accuracies)),
        "sigmas": np.asarray(sorted(args.sigmas, reverse=True)),
    }


def load_curves(path: Path, args: argparse.Namespace) -> dict | None:
    """The cached curves at `path` if they were simulated with these settings, else None.

    The sweep is the slow part — a few hundred utterances through every t of every curve —
    while restyling the figure is not, so the curves are kept next to the figure and
    reused whenever every setting they depend on is unchanged.
    """
    if not path.exists():
        return None
    data = dict(np.load(path))
    for key, want in settings(args).items():
        if key not in data or not np.array_equal(data[key], want):
            print(f"  cache {path} was made with a different {key}; resimulating")
            return None
    return data


def simulate(args: argparse.Namespace) -> dict:
    """Run both sweeps: the arrays the figure is drawn from, plus the settings behind them."""
    tokens, vocab_size = load_tokens(args)
    ts = np.linspace(0.0, 1.0, args.num_t)
    beta, gamma = make_schedule("linear_up"), make_schedule("linear_down")

    def scaled(sigma: float) -> "object":
        return make_schedule({"type": "linear_down", "scale": sigma})

    def report(name: str, curve: "object") -> None:
        print(f"  {name:28s} P_e(0)={curve[0]:.3f}  P_e(0.5)={curve[len(ts) // 2]:.3f}")

    print(f"sweeping {tokens.shape[0]} utts, V={vocab_size}, {args.num_t} points of t")
    # The baseline is the linear recipe as it trains: no mu AND unit noise, not a
    # sigma-scaled path with its third term removed.
    linear = error_curve(tokens, vocab_size, ts, scaled(1.0), beta, None, 1.0, args.seed)
    report("linear (no mu, sigma 1)", linear)
    # Ordered so the ramp reads the same way on both panels: from least to most of mu
    # showing through, i.e. accuracy ascending and sigma descending.
    by_accuracy = {}
    for acc in sorted(args.accuracies):
        by_accuracy[acc] = error_curve(
            tokens, vocab_size, ts, scaled(args.sigma), beta, gamma, acc, args.seed
        )
        report(f"sigma {args.sigma:g}, mu accuracy {acc:.0%}", by_accuracy[acc])
    by_sigma = {}
    for sigma in sorted(args.sigmas, reverse=True):
        by_sigma[sigma] = error_curve(
            tokens, vocab_size, ts, scaled(sigma), beta, gamma, args.mu_accuracy, args.seed
        )
        report(f"mu accuracy {args.mu_accuracy:.0%}, sigma {sigma:g}", by_sigma[sigma])
    return {
        **settings(args),
        "vocab_size": np.asarray(vocab_size),
        "ts": ts,
        "linear": linear,
        "accuracies": np.asarray(list(by_accuracy)),
        "by_accuracy": np.stack(list(by_accuracy.values())),
        "sigmas": np.asarray(list(by_sigma)),
        "by_sigma": np.stack(list(by_sigma.values())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweeps and write the figure."""
    args = parse_args(argv)
    for name, values in (("accuracies", args.accuracies), ("sigmas", args.sigmas)):
        if len(values) > len(MU_RAMP):
            print(f"at most {len(MU_RAMP)} {name} (the ramp has that many validated steps)")
            return 1

    cache = args.out.with_suffix(".npz")
    curves = None if args.resimulate else load_curves(cache, args)
    if curves is None:
        curves = simulate(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **curves)
        print(f"  cached the curves in {cache}")
    else:
        print(f"  curves from {cache} (--resimulate to rerun the sweep)")
    ts, linear = curves["ts"], curves["linear"]
    by_accuracy = dict(zip(curves["accuracies"].tolist(), curves["by_accuracy"], strict=True))
    by_sigma = dict(zip(curves["sigmas"].tolist(), curves["by_sigma"], strict=True))

    font = pick_font(args.font)
    style(font)
    print(f"  font: {font} (text and math)")
    # --width is the width of one panel, so the figure is twice as wide rather than two
    # squeezed halves: each panel has to stay readable on its own.
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(args.width * 2, args.width * 0.74), sharey=True
    )
    draw_curves(
        left,
        ts,
        linear,
        by_accuracy,
        label=lambda acc: f"{acc:.0%}",
        # Plain Unicode rather than mathtext, so the symbols take the title's own font.
        title=f"Varying \u03bc accuracy (\u03c3 = {args.sigma:g})",
        legend_title="$\\mu$ accuracy",
    )
    draw_curves(
        right,
        ts,
        linear,
        by_sigma,
        label=lambda sigma: f"{sigma:g}",
        title=f"Varying \u03c3 (\u03bc accuracy {100 * args.mu_accuracy:.0f}%)",
        legend_title="$\\sigma$",
        chosen=args.sigma,
    )
    left.set_ylabel("decoding error $P_e(t)$")

    fig.tight_layout(pad=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    png = args.out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    print(f"\nwrote {args.out}  and  {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
