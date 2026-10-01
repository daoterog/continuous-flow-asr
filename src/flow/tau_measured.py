"""Tau from the denoiser's own error, read off the binned metric it already reports."""

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
    """Builds tau from what the denoiser actually recovers, per time bin.

    The corruption sweep only knows what the interpolant destroys. This reads the model, so
    it accounts for the audio the denoiser is about to see: at low t the field carries
    nothing and recovery is whatever the audio alone supports, which is well above chance
    and nothing a model-free sweep can predict.

    It needs no mu and no datamodule — the numbers are already in the metric bank — so it
    refreshes on a linear run too.
    """

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[Callable | None, str | None]:
        """Read the denoiser's per-bin recovery and invert it. `datamodule` is unused."""
        key = f"{self.refresh_split}_denoiser_perr"
        if key not in model.metrics:
            return None, f"tau refresh skipped: no {key} metric to read"
        metric = model.metrics[key]
        if not bool(metric.seen().any()):
            return None, f"tau refresh skipped: {key} saw no samples this epoch"
        return self._tau_from(metric, model.vocab_size), None

    def _tau_from(self, metric: Metric, vocab_size: int) -> Callable:
        """Interpolate the populated bins onto a fixed grid, then invert.

        The fixed grid is what lets successive refreshes be compared at all: the bins move
        with every refresh, so a curve indexed by bin would not be the same curve twice.
        `bin_centers` gives the mean t actually seen in each bin rather than its midpoint,
        which matters because schedule bins differ in width by an order of magnitude.

        `p_e0` is the curve's own worst error, not 1 - 1/V: the denoiser hears the audio
        even at t=0, so it is not at chance there, and normalising by chance would claim a
        dead zone the model does not have.
        """
        seen = metric.seen().detach().cpu().numpy()
        ts = metric.bin_centers().numpy()[seen]
        rec = metric.recovery().detach().cpu().numpy()[seen]
        if rec.max() < 0.05:
            raise ValueError(
                f"the model recovers at most {rec.max():.3f} of tokens in any bin; tau "
                f"would collapse onto t=1. Train further first."
            )
        grid = np.linspace(0.0, 1.0, int(self.cfg.num_bins) + 1)
        on_grid = np.interp(
            grid,
            np.concatenate([ts, [1.0]]),
            np.concatenate([rec, [1.0]]),
            left=float(rec[0]),
        )
        return make_tau_inverse(
            grid.tolist(), on_grid.tolist(), vocab_size, p_e0=float(1.0 - on_grid[0])
        )
