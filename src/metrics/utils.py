"""Building and inspecting the metric bank: construction, naming, validation."""

import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

from .base import TimeBinnedMetric
from .loss import BinnedLoss
from .token_error import BinnedTokenError
from .wer import BinnedWER

if TYPE_CHECKING:
    from transformers import WhisperTokenizerFast

    from src.data import VocabMap
    from src.flow.schedule import TimeReparametrization

SPLITS = ("train", "val", "test")

# Decoding a batch to text roughly doubles the step cost, so WER is off on train by
# default; loss and P_e are scatter-adds and stay on everywhere.
DEFAULT_WER_BINS = {"train": 0, "val": 10, "test": 10}

# Groups with no axis to resolve, for either of two reasons.
#
#   mu        its forward takes audio and nothing else, so the same utterance yields the
#             same logits whichever bin its batch happened to draw — binning mu reports
#             how the draws landed, not anything about mu.
#   boundary  distillation's boundary rows are pinned at (s, t) = (0, 1), so every row is
#             at the same point and there is no spread to resolve.
#
# These get a single bin, which makes `overall()` the entire quantity and suppresses the
# per-bin histogram, the low/high pair and the bin-edge summary, all of which would be
# either constant or noise.
UNBINNED = ("mu", "boundary")


def bins_for(name: str, n: int) -> int:
    """Bin count for one group's metric: one bin if it has no axis, else `n`.

    A zero stays zero — that is how the config switches a metric off entirely.
    """
    return 1 if (n and name in UNBINNED) else n


def is_binned(metric: TimeBinnedMetric) -> bool:
    """Whether this metric resolves in t. A single bin is a scalar wearing the same API."""
    return getattr(metric, "num_bins", 1) > 1


def make_metrics(
    names: tuple[str, ...],
    tokenizer: "WhisperTokenizerFast",
    pad_id: int,
    vocab_map: "VocabMap | None" = None,
    loss_bins: int = 10,
    perr_bins: int = 10,
    wer_bins: dict | None = None,
    input_perr_bins: int = 0,
    time_reparam: "TimeReparametrization | None" = None,
    axes: dict[str, str] | None = None,
) -> nn.ModuleDict:
    """Build the `{split}_{group}_{metric}` ModuleDict the reporting mixin drives.

    Pass the SAME `time_reparam` the module draws t through. The bins then follow its
    quantiles instead of being equal-width in t, which is the difference
    between every bin holding ~1/n of the epoch and two thirds of them being empty.

    `axes` names what each group bins on, defaulting to "t". Flow-map distillation bins
    its off-diagonal group on the size of the jump, and the name reaches the logged key
    from here, so a gap is never reported under a key that says `t`. It also decides which
    groups `time_reparam` reaches: its edges are quantiles OF t, so they mean nothing on
    any other axis, and an axis that is already uniform keeps its equal-width bins.

    A ModuleDict keyed by split gives each split its own instances; sharing one would
    pool train and val into the same bins.
    """
    wer_bins = DEFAULT_WER_BINS | (wer_bins or {})
    axes = axes or {}

    def edges_for(n: int, axis: str) -> torch.Tensor | None:
        if axis != "t" or time_reparam is None or time_reparam.is_identity:
            return None
        return time_reparam.bin_edges(n)

    metrics = nn.ModuleDict()
    for split in SPLITS:
        if input_perr_bins:
            # argmax of the interpolant itself, with no model involved. Read against
            # denoiser_perr it separates what the field already carried from what the
            # denoiser added, which is the only way to see mu's effect on the input.
            input_axis = axes.get("input", "t")
            metrics[f"{split}_input_perr"] = BinnedTokenError(
                input_perr_bins,
                edges=edges_for(input_perr_bins, input_axis),
                pad_id=pad_id,
                axis=input_axis,
            )
        for name in names:
            # A group with no axis collapses to one bin; edges_for is then pointless, so
            # it is not called and the metric keeps the default full-range bin.
            binned = name not in UNBINNED
            axis = axes.get(name, "t")

            if n := bins_for(name, loss_bins):
                metrics[f"{split}_{name}_loss"] = BinnedLoss(
                    n, edges=edges_for(n, axis) if binned else None, axis=axis
                )
            if n := bins_for(name, perr_bins):
                metrics[f"{split}_{name}_perr"] = BinnedTokenError(
                    n, edges=edges_for(n, axis) if binned else None, pad_id=pad_id, axis=axis
                )
            if n := bins_for(name, wer_bins.get(split, 0)):
                metrics[f"{split}_{name}_wer"] = BinnedWER(
                    tokenizer, pad_id, vocab_map, num_bins=n,
                    edges=edges_for(n, axis) if binned else None, axis=axis,
                )
    return metrics


def split_key(key: str) -> tuple[str, str, str]:
    """`val_denoiser_wer` -> ("val", "denoiser", "wer").

    Safe as a plain split because no split, group or metric name contains an underscore;
    make_metrics is the only place these keys are built.
    """
    split, group, name = key.split("_", 2)
    return split, group, name


def section_for(key: str) -> str:
    """`val_denoiser_wer` -> "denoiser_wer", the wandb section the panels land in.

    One section per metric per group, holding train and val side by side. Grouping by
    split instead would put every metric of a split together and force a cross-section
    jump to compare the two.
    """
    _, group, name = split_key(key)
    return f"{group}_{name}"


def bins_key(section: str, split: str, metric: TimeBinnedMetric) -> str:
    """The wandb key one metric's per-bin histogram is logged under.

    The axis is in the key unless it is `t`, which keeps `denoiser_perr/val_bins` exactly
    where it has always been — existing runs are compared against that key — while a gap
    histogram lands at `offdiag_perr/train_bins_gap` and cannot be mistaken for one in t.

    Built here rather than inline because two callers must agree on the string: the one
    that logs the histogram and the one that declares its step metric. A drift between
    them loses the epoch axis silently.
    """
    if metric.axis == "t":
        return f"{section}/{split}_bins"
    return f"{section}/{split}_bins_{metric.axis}"


def edge_values(metric: TimeBinnedMetric) -> tuple[float, float] | tuple[None, None]:
    """The metric in its lowest and highest *populated* bin.

    The two ends of an axis answer different questions. In t: the low bin is where x_t is
    still noise and only the audio can carry the prediction, the high bin is where the
    target is nearly clean. In the distillation gap: the low bin is the short jump a
    teacher already handles, the high bin is the long jump the flow map exists to make, so
    the pair is also the collapse signature — low rising toward high is a student going
    constant in its endpoint. A single `overall` averages them and hides which end moved.

    Empty bins are skipped — under uniform bins with a reparameterized t the low ones are
    empty by construction.
    """
    seen = metric.seen()
    if not bool(seen.any()):
        return None, None
    values = metric.compute()
    populated = torch.nonzero(seen).flatten()
    return values[populated[0]].item(), values[populated[-1]].item()


def check_metrics(
    metrics: nn.ModuleDict, time_reparam: "TimeReparametrization | None" = None
) -> None:
    """Every leaf must declare `inputs`, and the bins must match how t is drawn."""
    uniform = []
    for key, m in metrics.items():
        if not hasattr(m, "inputs"):
            raise TypeError(
                f"{key} ({type(m).__name__}) has no `inputs` — declare it, e.g. "
                f"inputs = ('pred_ids', 'target_ids', 'bin_value')"
            )
        # Two cases where schedule edges are not something a metric is missing, and the
        # warning below would be noise: an unbinned metric has one bin spanning everything,
        # and a metric on any axis other than t is binned on a quantity tau says nothing
        # about — the distillation gap is drawn uniform on [0, 1], so equal-width bins are
        # already what tau's edges exist to achieve.
        reparametrized = time_reparam is not None and not time_reparam.is_identity
        on_t = getattr(m, "axis", "t") == "t"
        if reparametrized and on_t and is_binned(m) and not getattr(m, "schedule_bins", False):
            uniform.append(key)
    if uniform:
        warnings.warn(
            f"t is drawn through {time_reparam!r}, but {len(uniform)} metric(s) use "
            f"equal-width t bins — most of those bins will stay empty. Pass the same "
            f"time_reparam to make_metrics(). ({', '.join(uniform[:4])}"
            f"{', ...' if len(uniform) > 4 else ''})",
            stacklevel=2,
        )
