"""Tau from the denoiser's own error, read off a curve the schedule does not shape."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from torchmetrics import Metric

from src.flow.schedule import make_tau_inverse
from src.flow.tau_base import TauScheduler

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


class MeasuredTau(TauScheduler):
    """Builds tau from what the denoiser actually recovers, per uniform time bin.

    The corruption sweep only knows what the interpolant destroys. This reads the model, so
    it accounts for the audio the denoiser is about to see: at low t the field carries
    nothing and recovery is whatever the audio alone supports, which is well above chance
    and nothing a model-free sweep can predict.

    It reads `{split}_denoiser_uperr` — uniform bins, never re-cut — rather than the
    schedule-binned `denoiser_perr`. Those bins are tau's own quantiles, so that curve can
    only report where tau already draws, and tau draws nothing below its t0. Feeding it
    back closed a loop: the region below t0 went unmeasured, `make_tau_inverse`
    extrapolated it flat from the lowest bin, and normalising by that curve's worst error
    moved t0 further right every refresh — observed as 0.250 -> 0.500 -> 0.625 -> 0.750,
    then stuck, with three quarters of the axis neither trained nor measured.

    `anchor_zero` closes the other half: tau^-1(0) == 0, so the schedule keeps drawing low
    t and the uniform metric keeps having something to report there.

    It needs no mu, so it refreshes on a linear run too.
    """

    @property
    def metric_bins(self) -> int:
        """Bins for the uniform curve. Finer than the diagnostic perr, and independent."""
        return int(self.cfg.num_bins)

    @property
    def min_recovery(self) -> float:
        """Recovery the best bin must reach before its curve is worth inverting."""
        return float(self.cfg.get("measured_min_recovery", 0.05))

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[Callable | None, str | None]:
        """Invert the uniform P_e curve. `datamodule` is unused, the numbers are in the metric."""
        key = f"{self.refresh_split}_denoiser_tau_perr"
        if key not in model.metrics:
            return None, f"tau refresh skipped: no {key} metric to read"
        metric = model.metrics[key]
        if not bool(metric.seen().any()):
            return None, f"tau refresh skipped: {key} saw no samples this epoch"

        # A model too weak to anchor a curve is the normal state at the first refresh, not
        # an error: tau would collapse onto t=1 and starve every t below it. Keep the
        # schedule it has and look again next time, rather than killing the run.
        best = float(metric.recovery()[metric.seen()].max())
        if best < self.min_recovery:
            return None, (
                f"tau refresh skipped: the model recovers at most {best:.3f} of tokens in "
                f"any bin, under the {self.min_recovery} needed to anchor tau"
            )
        return self._tau_from(metric, model.vocab_size), None

    def _tau_from(self, metric: Metric, vocab_size: int) -> Callable:
        """Invert the populated bins of the uniform curve.

        `bin_centers` gives the mean t actually seen in each bin rather than its midpoint.
        With uniform edges the two nearly agree, unlike under schedule bins whose widths
        differ by an order of magnitude.

        `p_e0` is the error in the lowest populated bin, not chance: the denoiser hears the
        audio even at t=0, so it is nowhere near chance there, and normalising by chance
        would claim a dead zone the model does not have.
        """
        seen = metric.seen().detach().cpu().numpy()
        ts = metric.bin_centers().numpy()[seen]
        rec = metric.recovery().detach().cpu().numpy()[seen]
        # grid = np.linspace(0.0, 1.0, int(self.cfg.num_bins) + 1)
        # on_grid = np.interp(
        #     grid,
        #     np.concatenate([ts, [1.0]]),
        #     np.concatenate([rec, [1.0]]),
        #     left=float(rec[0]),
        # )
        # Carry the lowest reading down to t=0 when the bins below it are still empty.
        # The first refresh inherits the prior's coverage, so there is genuinely nothing
        # measured down there yet; without this the curve starts at the lowest populated
        # bin and `anchor_zero` would anchor to that instead of to 0. Stating it flat is
        # an assumption, but `anchor_zero` then makes the schedule draw there and the next
        # refresh replaces it with a measurement — self-correcting, where reading only the
        # populated window ratcheted t0 upward.
        if ts[0] > 0.0:
            ts = np.concatenate([[0.0], ts])
            rec = np.concatenate([[rec[0]], rec])
        return make_tau_inverse(
            ts, rec, vocab_size, p_e0=float(1.0 - rec[0]), anchor_zero=True
        )
