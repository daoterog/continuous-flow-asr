"""Token error rate binned by flow time — the quantity the tau schedule is built from."""

import torch

from .base import Edges, TimeBinnedMetric


class BinnedTokenError(TimeBinnedMetric):
    """Token error rate P_e(t) per t-bin — the quantity the time reparameterization needs.

    Deliberately token-level, not WER. Eq. 25's |V|/(|V|-1) factor is exactly 1/P_e(0),
    which only holds when P_e(0) = (V-1)/V, the hit rate of an argmax over pure noise.
    WER has no such anchor (insertions push it past 100% on an untrained model), so
    feeding BinnedWER into the reparameterization would drive tau negative.

    Cheap — a comparison and two sums, no decoding — so it can run every epoch and let
    tau be re-estimated from the model rather than from the model-free corruption sweep.
    """

    inputs = ("pred_ids", "target_ids", "t")
    label = r"$P_e$ (token error)"
    ylim = (0, 1.02)

    def __init__(self, num_bins: int = 100, edges: Edges = None, **kwargs: object) -> None:
        """Track a running wrong-token count and total per t-bin."""
        super().__init__(num_bins=num_bins, edges=edges, **kwargs)
        self.add_state("errors", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")

    def update(
        self, pred_ids: torch.Tensor, target_ids: torch.Tensor, t: torch.Tensor
    ) -> None:
        """Accumulate one batch of (B, L) predictions against (B, L) targets."""
        idx = self._bin_indices(t)
        wrong = (pred_ids != target_ids).sum(dim=1).float()
        total = torch.full_like(wrong, target_ids.size(1))
        self.errors.scatter_add_(0, idx, wrong)
        self.count.scatter_add_(0, idx, total)

    def compute(self) -> torch.Tensor:
        """(num_bins,) P_e; empty bins report 0."""
        return self.errors / self.count.clamp(min=1)

    def seen(self) -> torch.Tensor:
        """(num_bins,) bool — which bins actually received a sample."""
        return self.count > 0

    def overall(self) -> float:
        """P_e across every bin, weighted by how many tokens landed in each."""
        n = self.count.sum()
        if n == 0:
            return float("nan")
        return (self.errors.sum() / n).item()

    def recovery(self) -> torch.Tensor:
        """(num_bins,) 1 - P_e, i.e. token recovery — what make_tau_inverse consumes."""
        return 1 - self.compute()
