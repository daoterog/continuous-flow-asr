"""How the learned source's preset choices spread: the share of positions on each sigma."""

from collections.abc import Sequence

import numpy as np
import torch

from .base import TimeBinnedMetric


def preset_edges(scales: Sequence[float]) -> torch.Tensor:
    """Bin edges that put each preset sigma in a bin of its own.

    The boundaries are the geometric midpoints between neighbours, extended past both ends
    by the same ratio, so a geometric grid gets bins of equal width on a log axis.
    """
    s = np.asarray(scales, dtype=float)
    if len(s) == 1:
        return torch.tensor([s[0] * 0.9, s[0] * 1.1], dtype=torch.float32)
    mids = np.sqrt(s[1:] * s[:-1])
    return torch.tensor(
        np.concatenate([[s[0] ** 2 / mids[0]], mids, [s[-1] ** 2 / mids[-1]]]),
        dtype=torch.float32,
    )


class BinnedShare(TimeBinnedMetric):
    """The share of samples in each bin: a histogram of the binned value itself.

    Binned on each real position's chosen sigma with `preset_edges`, it is how the learned
    source spreads over its presets, and `src.reporting.log_metrics` logs it per epoch like
    every other binned metric. The counts are the base class's own, so updating costs one
    scatter-add. `overall` is the mean of the values binned: the mean sigma.
    """

    inputs = ("bin_value",)
    label = "share"
    ylim = (0, 1)

    def update(self, bin_value: torch.Tensor) -> None:
        """Count one batch of values into their bins."""
        self._bin_indices(bin_value.detach())

    def compute(self) -> torch.Tensor:
        """(num_bins,) the share of all values counted that fell in each bin."""
        return self.bin_value_count / self.bin_value_count.sum().clamp(min=1)

    def seen(self) -> torch.Tensor:
        """(num_bins,) every bin: an empty one is a real zero share, not a missing value."""
        return torch.ones_like(self.bin_value_count, dtype=torch.bool)

    def overall(self) -> float:
        """The mean of every value counted."""
        n = self.bin_value_count.sum()
        if n == 0:
            return float("nan")
        return (self.bin_value_sum.sum() / n).item()
