"""Tau from the interpolant alone, with mu stood in at its measured accuracy."""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from omegaconf import DictConfig
from torchmetrics import Metric

from src.flow.schedule import TimeReparametrization
from src.flow.tau_base import TauScheduler

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


class SimulatedTau(TauScheduler):
    """Integrates how much of the target survives the interpolant, with no denoiser in it.

    The curve is what corruption alone leaves recoverable, which is why it is available
    before a single training step. What makes it move during a run is mu — the stand-in is
    right with probability `mu_accuracy`, so as mu learns the audio the dead zone shrinks
    and tau follows. Without a gamma term nothing moves at all, and `needs_mu` keeps those
    runs from refreshing a curve that cannot have changed.

    That accuracy is the strategy's one piece of state, averaged across epochs so tau
    tracks mu's trend rather than SGD wobble.
    """

    needs_mu = True

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """Start with no reading of mu; `observe_mu` supplies them as epochs finish."""
        super().__init__(cfg, seed=seed)
        self.mu_accuracy: float | None = None
        self._ema = 0.0
        self._ema_steps = 0

    def observe_mu(self, metrics: Mapping[str, Metric]) -> None:
        """Fold mu's error on the refresh split into the running accuracy.

        Called at the end of every training epoch, not only when a refresh is due, so the
        average sees every reading it can and the momentum's lag counts epochs.
        """
        key = f"{self.refresh_split}_mu_perr"
        if key not in metrics:
            return
        perr = float(metrics[key].overall())
        if math.isnan(perr):
            return
        self.smooth_mu_accuracy(1.0 - perr)

    def smooth_mu_accuracy(self, measured: float) -> float:
        """Bias-corrected EMA of mu's accuracy, kept on `mu_accuracy` for the next refresh.

        The correction divides by the weight the real readings carry so far, `1 - m**k`.
        Without it the average starts at the 0.0 seed, which is a value mu never reported:
        at momentum 0.8 a constant 0.70 would read 0.14 on the first epoch.
        """
        m = float(self.cfg.get("mu_accuracy_momentum", 0.0))
        if not 0.0 <= m < 1.0:
            raise ValueError(f"schedule.mu_accuracy_momentum must be in [0, 1), got {m}")
        if m == 0.0:
            self.mu_accuracy = measured
            return measured
        self._ema = m * self._ema + (1.0 - m) * measured
        self._ema_steps += 1
        self.mu_accuracy = self._ema / (1.0 - m**self._ema_steps)
        return self.mu_accuracy

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[TimeReparametrization | None, str | None]:
        """Rebuild the exact interpolant curve at mu's current accuracy and invert it.

        `datamodule` goes unused — the curve depends on the vocabulary size and the
        coefficients, not on which tokens happen to be in a batch — but it stays in the
        signature because `MeasuredTau` needs it and the caller holds only a `TauScheduler`.
        """
        del datamodule
        if model.gamma_fn is not None and self.mu_accuracy is None:
            return None, "tau refresh skipped: the simulated curve needs mu's accuracy"
        return TimeReparametrization.exact(
            model.vocab_size,
            alpha_fn=model.alpha_fn,
            beta_fn=model.beta_fn,
            gamma_fn=model.gamma_fn,
            mu_accuracy=0.0 if self.mu_accuracy is None else self.mu_accuracy,
        ), None
