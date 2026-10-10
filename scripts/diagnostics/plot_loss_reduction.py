"""Figure: how much each token counts in the objective, per pad setting and reduction.

`flow.loss_reduction` and `flow.pad_loss_weight`/`pad_decay` decide together how the
batch's per-token cross-entropy becomes one number. This draws, on real train.100
transcripts and the weights `pl_module.token_weights` actually produces:

  (a) the weight of a position against its distance from the end of the transcript —
      what `pad_decay` does to the pads.
  (b) how much one real token counts, against its utterance's length, relative to a plain
      mean over the canvas (1.0 = every position counted once). Under `sequence` each row
      is normalised by its own weight total, so short utterances' tokens count more.
      Under `token` the whole batch shares one denominator, so every real token counts
      the same however long its utterance is.
  (c) the share of the objective that goes to padding at all.

At pad_loss_weight 1.0 the two reductions coincide, because every row is canvas_len long;
that row is drawn as the reference.

    python -m scripts.diagnostics.plot_loss_reduction
    python -m scripts.diagnostics.plot_loss_reduction --batch 128 --n-utts 4096
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import WhisperTokenizerFast

from src.constants import data_root_for
from src.data import AudioFeatureDataModule
from src.flow_matching import token_weights

# Categorical slots 1-3 of the validated palette, in fixed order, one per pad setting.
# The reduction is the second encoding: solid for `sequence`, dashed for `token`.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"

# (label, pad_loss_weight, pad_decay)
SETTINGS = [
    ("pad 1.0 (today)", 1.0, 0.0),
    ("pad 0.1 flat", 0.1, 0.0),
    ("pad 0.1, decay 8", 0.1, 8.0),
]
REDUCTIONS = ("sequence", "token")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which transcripts, and how they are batched."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--config", default="clean")
    p.add_argument("--split", default="train.100")
    p.add_argument("--n-utts", type=int, default=4096)
    p.add_argument("--batch", type=int, default=128, help="training batch size")
    p.add_argument("--canvas-len", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("figures/loss_reduction.png"))
    return p.parse_args(argv)


def load_tokens(args: argparse.Namespace) -> tuple[torch.Tensor, int]:
    """(n_utts, canvas_len) targets and the pad id they are padded with."""
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
    return dm.sample_tokens("train", args.n_utts, seed=args.seed), int(dm.pad_id)


def shares(weights: torch.Tensor, reduction: str) -> torch.Tensor:
    """(B, L) each position's share of the objective, scaled so a plain mean gives 1.

    The objective is linear in the per-token losses, so its gradient with respect to one
    token's loss is exactly that token's coefficient. These are those coefficients.
    """
    b, length = weights.shape
    if reduction == "sequence":
        coef = weights / (b * weights.sum(dim=1, keepdim=True))
    else:
        coef = weights / weights.sum()
    return coef * b * length


def main(argv: Sequence[str] | None = None) -> None:
    """Draw the three panels and write a PNG and a PDF."""
    args = parse_args(argv)
    tokens, pad_id = load_tokens(args)
    batches = tokens[: len(tokens) // args.batch * args.batch].split(args.batch)
    n_real = (tokens != pad_id).sum(dim=1)
    print(
        f"{len(tokens)} utterances, {len(batches)} batches of {args.batch}; real tokens "
        f"median {int(n_real.median())}, range {int(n_real.min())}-{int(n_real.max())} "
        f"of {args.canvas_len}"
    )

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
                         "axes.edgecolor": GRID})
    fig, (ax_w, ax_real, ax_pad) = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")

    # (a) weight against distance from the end of the transcript.
    distance = np.arange(-10, 61)
    row = torch.zeros(1, args.canvas_len, dtype=torch.long)
    end = 60
    row[0, end:] = pad_id
    row[0, :end] = pad_id + 1
    for color, (label, floor, decay) in zip(SERIES, SETTINGS):
        w = token_weights(row, pad_id, floor, decay)
        w = torch.ones_like(row, dtype=torch.float) if w is None else w
        ax_w.plot(distance, w[0, end + distance - 1].numpy(), color=color, lw=2, label=label)
    ax_w.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
    ax_w.text(14, 0.72, "first pad = the stop decision;\nonly decay keeps it at full weight",
              color=INK_MUTED,
              fontsize=8)
    ax_w.set(xlabel="position relative to the last real token", ylabel="weight w",
             title="(a) per-position weight", ylim=(0, 1.08))

    # (b) and (c), measured over real batches.
    pad_share: dict[tuple[str, str], float] = {}
    for color, (label, floor, decay) in zip(SERIES, SETTINGS):
        for reduction in REDUCTIONS:
            if floor == 1.0 and reduction == "token":
                continue  # identical to `sequence`: every row is canvas_len long
            lengths, per_real, pads, totals = [], [], [], []
            for batch in batches:
                w = token_weights(batch, pad_id, floor, decay)
                w = torch.ones_like(batch, dtype=torch.float) if w is None else w
                s = shares(w, reduction)
                is_pad = batch == pad_id
                lengths.append((~is_pad).sum(dim=1))
                # Real tokens all weigh 1, so every real token in a row has one share.
                per_real.append(s.masked_fill(is_pad, 0).amax(dim=1))
                pads.append((s * is_pad).sum())
                totals.append(s.sum())
            lengths = torch.cat(lengths).numpy()
            per_real = torch.cat(per_real).numpy()
            pad_share[(label, reduction)] = float(sum(pads) / sum(totals))

            edges = np.arange(0, lengths.max() + 8, 8)
            mids, means = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                pick = (lengths >= lo) & (lengths < hi)
                if pick.sum() >= 5:
                    mids.append((lo + hi) / 2)
                    means.append(per_real[pick].mean())
            name = label if floor == 1.0 else f"{label}, {reduction}"
            ax_real.plot(mids, means, color=color, lw=2,
                         ls="-" if reduction == "sequence" else "--", label=name)
    ax_real.set(xlabel="real tokens in the utterance",
                ylabel="weight of one real token (plain mean = 1)",
                title="(b) how much one real token counts")
    ax_real.set_yscale("log")
    ax_real.set_yticks([1, 1.5, 2, 3, 4, 6], ["1", "1.5", "2", "3", "4", "6"])
    ax_real.minorticks_off()

    labels = [f"{label.replace(' (today)', '').replace(', ', ',\n')}\n{reduction}"
              for (label, reduction) in pad_share]
    colors = [SERIES[[s[0] for s in SETTINGS].index(label)] for (label, _) in pad_share]
    hatches = ["" if r == "sequence" else "//" for (_, r) in pad_share]
    bars = ax_pad.bar(range(len(pad_share)), list(pad_share.values()), color=colors,
                      edgecolor="white", linewidth=2, hatch=hatches)
    for bar, v in zip(bars, pad_share.values()):
        ax_pad.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.0%}",
                    ha="center", color=INK, fontsize=8)
    ax_pad.set_xticks(range(len(pad_share)), labels, fontsize=7)
    ax_pad.set(ylabel="share of the objective on padding", ylim=(0, 1),
               title="(c) how much of the loss is padding")

    for ax in (ax_w, ax_real, ax_pad):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_w.legend(frameon=False, fontsize=8)
    ax_real.legend(frameon=False, fontsize=7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")
    for (label, reduction), v in pad_share.items():
        print(f"  {label:18s} {reduction:8s} padding share {v:.1%}")


if __name__ == "__main__":
    main()
