"""Building and inspecting the metric bank: construction, naming, validation."""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn

from .base import TimeBinnedMetric, schedule_bin_edges
from .loss import BinnedLoss
from .token_error import BinnedTokenError
from .wer import BinnedWER

if TYPE_CHECKING:
    from transformers import WhisperTokenizerFast

    from src.data import VocabMap

SPLITS = ("train", "val", "test")

# Decoding a batch to text roughly doubles the step cost, so WER is off on train by
# default; loss and P_e are scatter-adds and stay on everywhere.
DEFAULT_WER_BINS = {"train": 0, "val": 10, "test": 10}

# Models whose prediction does not depend on t. mu_psi's forward takes audio and nothing
# else, so the same utterance yields the same logits whichever t-bin its batch happened to
# draw — binning mu reports how the t draws landed, not anything about mu. These get a
# single bin, which makes `overall()` the entire quantity and suppresses the per-bin
# histogram, the low_t/high_t pair and the bin-edge summary, all of which would be either
# constant or noise. The denoiser and the interpolant do vary with t and stay binned.
AUDIO_ONLY = ("mu",)


def bins_for(name: str, n: int) -> int:
    """Bin count for one model's metric: one bin if it cannot vary with t, else `n`.

    A zero stays zero — that is how the config switches a metric off entirely.
    """
    return 1 if (n and name in AUDIO_ONLY) else n


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
    tau_bins: int = 0,
    sample_t_fn: Callable | None = None,
) -> nn.ModuleDict:
    """Build the `{split}_{model}_{metric}` ModuleDict the reporting mixin drives.

    Pass the SAME sample_t_fn the module samples t with. The bins then follow the
    schedule's quantiles instead of being equal-width in t, which is the difference
    between every bin holding ~1/n of the epoch and two thirds of them being empty.

    A ModuleDict keyed by split gives each split its own instances; sharing one would
    pool train and val into the same bins.
    """
    wer_bins = DEFAULT_WER_BINS | (wer_bins or {})

    def edges_for(n: int) -> torch.Tensor | None:
        return None if sample_t_fn is None else schedule_bin_edges(sample_t_fn, n)

    metrics = nn.ModuleDict()
    for split in SPLITS:
        if input_perr_bins:
            # argmax of the interpolant itself, with no model involved. Read against
            # denoiser_perr it separates what the field already carried from what the
            # denoiser added, which is the only way to see mu's effect on the input.
            metrics[f"{split}_input_perr"] = BinnedTokenError(
                input_perr_bins, edges=edges_for(input_perr_bins), pad_id=pad_id
            )
        for name in names:
            # An audio-only model collapses to one bin; edges_for is then pointless, so
            # it is not called and the metric keeps the default full-range bin.
            binned = name not in AUDIO_ONLY

            if n := bins_for(name, loss_bins):
                metrics[f"{split}_{name}_loss"] = BinnedLoss(
                    n, edges=edges_for(n) if binned else None
                )
            # The curve a measured tau refresh reads. Uniform and never re-cut, so it
            # reports where the schedule is NOT drawing as well as where it is — the
            # schedule-binned perr below can only ever describe tau's own quantiles.
            if tau_bins and binned:
                metrics[f"{split}_{name}_tau_perr"] = BinnedTokenError(
                    tau_bins, edges=None, pad_id=pad_id, fixed_edges=True
                )
            if n := bins_for(name, perr_bins):
                metrics[f"{split}_{name}_perr"] = BinnedTokenError(
                    n, edges=edges_for(n) if binned else None, pad_id=pad_id
                )
            if n := bins_for(name, wer_bins.get(split, 0)):
                metrics[f"{split}_{name}_wer"] = BinnedWER(
                    tokenizer, pad_id, vocab_map, num_bins=n,
                    edges=edges_for(n) if binned else None,
                )
    return metrics


def split_key(key: str) -> tuple[str, str, str]:
    """`val_denoiser_wer` -> ("val", "denoiser", "wer").

    Safe as a plain split because no split, model or metric name contains an underscore;
    make_metrics is the only place these keys are built.
    """
    split, model, name = key.split("_", 2)
    return split, model, name


def section_for(key: str) -> str:
    """`val_denoiser_wer` -> "denoiser_wer", the wandb section the panels land in.

    One section per metric per model, holding train and val side by side. Grouping by
    split instead would put every metric of a split together and force a cross-section
    jump to compare the two.
    """
    _, model, name = split_key(key)
    return f"{model}_{name}"


def edge_values(metric: TimeBinnedMetric) -> tuple[float, float] | tuple[None, None]:
    """The metric in its lowest and highest *populated* t bin.

    The two ends of the flow answer different questions: the low-t bin is where x_t is
    still noise and only the audio can carry the prediction, the high-t bin is where the
    target is nearly clean. A single `overall` averages them and hides which end moved.
    Empty bins are skipped — under uniform bins with a reparameterized t the low ones are
    empty by construction.
    """
    seen = metric.seen()
    if not bool(seen.any()):
        return None, None
    values = metric.compute()
    populated = torch.nonzero(seen).flatten()
    return values[populated[0]].item(), values[populated[-1]].item()


def check_metrics(metrics: nn.ModuleDict, sample_t_fn: Callable | None = None) -> None:
    """Every leaf must declare `inputs`, and the bins must match how t is drawn."""
    uniform = []
    for key, m in metrics.items():
        if not hasattr(m, "inputs"):
            raise TypeError(
                f"{key} ({type(m).__name__}) has no `inputs` — declare it, e.g. "
                f"inputs = ('pred_ids', 'target_ids', 't')"
            )
        # An unbinned metric has one bin spanning all of t, so schedule edges are not
        # something it is missing — the warning below would be noise.
        if sample_t_fn is not None and is_binned(m) and not getattr(m, "schedule_bins", False):
            uniform.append(key)
    if uniform:
        warnings.warn(
            f"t is drawn through sample_t_fn, but {len(uniform)} metric(s) use "
            f"equal-width t bins — most of those bins will stay empty. Pass the same "
            f"sample_t_fn to make_metrics(). ({', '.join(uniform[:4])}"
            f"{', ...' if len(uniform) > 4 else ''})",
            stacklevel=2,
        )
