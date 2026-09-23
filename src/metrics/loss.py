"""Cross-entropy binned by flow time."""

import torch

from .base import Edges, TimeBinnedMetric


class BinnedLoss(TimeBinnedMetric):
    """Mean per-sequence loss as a function of the flow time t, per bin.

    The companion to BinnedWER. WER says how much of the transcript came out right; this
    says how hard the objective was. Reading them together separates "the model cannot do
    this" from "x_t carried no information at this t" — which, given the corruption sweep,
    is the distinction that matters below t ~ 0.8.

    Unlike BinnedWER this costs nothing: no decoding, just a scatter-add, so it is worth
    tracking on training epochs as well as validation ones.
    """

    inputs = ("loss_per_seq", "t")
    label = "cross-entropy"
    ylim = (0, None)

    def __init__(self, num_bins: int = 10, edges: Edges = None, **kwargs: object) -> None:
        """Track a running loss sum and count per t-bin."""
        super().__init__(num_bins=num_bins, edges=edges, **kwargs)
        self.add_state("loss_sum", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")

    def update(self, loss_per_seq: torch.Tensor, t: torch.Tensor) -> None:
        """Accumulate one batch. loss_per_seq: (B,) per row, NOT the batch mean."""
        idx = self._bin_indices(t)
        loss = loss_per_seq.detach().float()
        self.loss_sum.scatter_add_(0, idx, loss)
        self.count.scatter_add_(0, idx, torch.ones_like(loss))

    def compute(self) -> torch.Tensor:
        """(num_bins,) mean loss; empty bins report 0."""
        return self.loss_sum / self.count.clamp(min=1)

    def seen(self) -> torch.Tensor:
        """(num_bins,) bool — which bins actually received a sample."""
        return self.count > 0

    def overall(self) -> float:
        """Mean loss across every bin, weighted by how many samples landed in each."""
        n = self.count.sum()
        if n == 0:
            return float("nan")
        return (self.loss_sum.sum() / n).item()
