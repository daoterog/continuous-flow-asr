"""Bin geometry: the t -> bin mapping every time-binned metric is built on.

Scoring a transcript is `src.metrics.scoring`; this file is only about where a sample's
t falls and how the bins are cut.
"""

import numpy as np
import torch
from torchmetrics import Metric

# torch tensor, numpy array or plain sequence — anything torch.as_tensor accepts.
Edges = torch.Tensor | np.ndarray | list[float] | None


class TimeBinnedMetric(Metric):
    """Base for metrics. All they share is the t -> bin mapping.

    Equal-width bins by default. Pass `edges` to bin some other way; in particular
    `TimeReparametrization.bin_edges(n)`, which is what keeps every bin populated once t
    is drawn through a reparameterization rather than uniformly.

    `num_bins=1` is the degenerate case, and is how a quantity that cannot vary with t is
    measured — mu_psi's forward takes audio and nothing else, so its prediction is the
    same whichever bin its batch drew. One bin makes `overall()` the whole quantity, and
    `metrics.utils.is_binned` reads False, which suppresses the histogram and the
    low_t/high_t pair downstream.
    """

    def __init__(self, num_bins: int = 10, edges: Edges = None, **kwargs: object) -> None:
        """Build the bin geometry; `edges` overrides `num_bins` when given."""
        super().__init__(**kwargs)
        if edges is None:
            edges = torch.linspace(0, 1, num_bins + 1)
            self.schedule_bins = False
        else:
            edges = torch.as_tensor(edges, dtype=torch.float32)
            if not bool((edges[1:] > edges[:-1]).all()):
                raise ValueError(f"bin edges must be strictly increasing, got {edges}")
            num_bins = edges.numel() - 1
            self.schedule_bins = True
        self.num_bins = num_bins
        self._edges = edges
        # Running sum and count of the t values seen per bin, so bin_centers can report
        # where the samples actually were rather than assuming they filled the bin evenly.
        self.add_state("t_sum", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("t_seen", default=torch.zeros(num_bins), dist_reduce_fx="sum")

    def _bin_indices(self, t: torch.Tensor) -> torch.Tensor:
        """Map t to the index of the bin [edges[k], edges[k+1]) it falls in.

        right=True so a t landing exactly on an edge goes to the bin above it, matching
        the half-open convention (and the floor that equal-width binning used to do).

        Records each t as it goes. Every subclass calls this exactly once per update, so
        accumulating here is what keeps `bin_centers` honest without each of them having
        to remember to do it.
        """
        t = t.float()
        idx = torch.bucketize(t, self._edges[1:-1].to(t.device), right=True)
        self.t_sum.scatter_add_(0, idx, t)
        self.t_seen.scatter_add_(0, idx, torch.ones_like(t))
        return idx

    def set_edges(self, edges: Edges) -> None:
        """Re-bin in place, keeping the same number of bins.

        Used when tau is refreshed mid-run: the schedule's quantiles move, so the bins
        have to move with them or they no longer hold ~1/n of the samples each. The bin
        count is fixed because the accumulator states are sized by it — call reset()
        first, since counts collected under the old edges do not describe the new ones.
        """
        edges = torch.as_tensor(edges, dtype=torch.float32)
        if edges.numel() - 1 != self.num_bins:
            raise ValueError(
                f"re-binning must keep {self.num_bins} bins, got {edges.numel() - 1}"
            )
        if not bool((edges[1:] > edges[:-1]).all()):
            raise ValueError(f"bin edges must be strictly increasing, got {edges}")
        self._edges = edges
        self.schedule_bins = True

    def bin_edges(self) -> torch.Tensor:
        """(num_bins + 1,) the bin boundaries in t."""
        return self._edges.detach().cpu()

    def bin_centers(self) -> torch.Tensor:
        """(num_bins,) the mean t seen in each bin, falling back to its midpoint.

        The midpoint is only the representative t if the samples fill the bin evenly, and
        under a schedule they do not: bins are quantiles of u, so t inside one is spread
        as tau^-1 across a uniform slice of u, and the widths differ by an order of
        magnitude — the lowest bin can span [0.19, 0.66] where the top ones span ~0.01.
        Attributing a bin's measured recovery to its midpoint then places it at a t the
        samples were never near, which matters because that pairing is what the measured
        tau is rebuilt from.

        Empty bins have no samples to average, so they keep the midpoint; before the first
        update that is every bin.
        """
        edges = self.bin_edges()
        midpoints = (edges[:-1] + edges[1:]) / 2
        seen = self.t_seen.detach().cpu()
        mean_t = self.t_sum.detach().cpu() / seen.clamp(min=1)
        return torch.where(seen > 0, mean_t, midpoints)
