"""Simulated until the denoiser is worth measuring, then measured. A handover, not a blend.

`MeasuredTau` already declines to invert a curve whose best bin is under
`measured_min_recovery`, and the caller keeps whatever tau it holds — which at the start of
a run is the exact interpolant curve `train.build_sample_t_fn` builds. So a handover of
sorts already happens. What it does not do is keep the held curve current: that prior is
built once, at `schedule.mu_accuracy` (0.0, because mu is at init), and it stays there for
the whole warmup.

That matters on a path with gamma, because the interpolant curve moves a long way as mu
learns. Measured on the closed form at V=1193, tau clears 1e-3 at

    mu accuracy 0.0 -> t = 0.199        0.5 -> t = 0.047        0.9 -> t = 0.029

so a warmup frozen at accuracy 0.0 trains the opening epochs as though mu carried nothing,
while mu is in fact the only thing carrying the low-t end. Refreshing it costs ~36 ms now
that the curve is integrated rather than swept, so there is no reason to leave it stale.

This strategy therefore runs the simulated curve at mu's *current* accuracy for as long as
the denoiser is too weak to measure, and switches to the measured curve once it is not.

The switch is sticky. A later probe that falls back under the bar means a bad draw or a
transient, not that the model forgot how to decode, and bouncing between two differently
normalised curves would move the schedule for a reason that has nothing to do with
progress. Once measured, always measured; the guards inside `MeasuredTau` still apply, and
a refresh it declines leaves the last good curve in place as before.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from omegaconf import DictConfig
from torchmetrics import Metric

from src.flow.tau_base import TauScheduler
from src.flow.tau_measured import MeasuredTau
from src.flow.tau_simulated import SimulatedTau

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


class WarmupTau(TauScheduler):
    """Hands over from the simulated curve to the measured one, once and in one direction.

    Both strategies are held, not reimplemented: each keeps its own guards, its own state
    and its own reasons for declining, and this only decides which one speaks.
    """

    # The simulated half wants mu, the measured half does not, and the measured half is
    # where the run ends up — so a linear run should still refresh. `is_due` reads this to
    # decide whether a run without gamma refreshes at all.
    needs_mu = False

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """Build both strategies and start on the simulated one."""
        super().__init__(cfg, seed=seed)
        self.simulated = SimulatedTau(cfg, seed=seed)
        self.measured = MeasuredTau(cfg, seed=seed)
        self.switched = False
        self.source = "simulated"

    @property
    def mu_accuracy(self) -> float | None:
        """Mu's smoothed accuracy, which only the simulated half tracks."""
        return self.simulated.mu_accuracy

    def observe_mu(self, metrics: Mapping[str, Metric]) -> None:
        """Keep feeding the simulated half every epoch, including after the switch.

        After the handover its curve is no longer used, but the average it maintains is
        what `tau/mu_accuracy_ema` reports, and a gap in that series would read as mu
        having stopped rather than as the schedule having moved on.
        """
        self.simulated.observe_mu(metrics)

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[Callable | None, str | None]:
        """Try the measured curve; fall back to a freshly built simulated one until it takes.

        The measured attempt runs first even during warmup, because its own recovery
        reading is what decides whether warmup is over — there is no cheaper test, and the
        probe is ~5 s against an epoch of ~43 s.
        """
        if self.switched:
            return self.measured.refresh(model, datamodule)

        tau_inverse, declined = self.measured.refresh(model, datamodule)
        if tau_inverse is not None:
            self.switched = True
            self.source = "measured"
            return tau_inverse, (
                f"tau warmup over at epoch {model.current_epoch}: the denoiser clears "
                f"{self.measured.min_recovery} recovery, switching to the measured curve"
            )

        # Still warming up. Rebuild the interpolant curve at mu's accuracy as of this
        # epoch rather than leaving the startup prior standing.
        fallback, why = self.simulated.refresh(model, datamodule)
        if fallback is None:
            return None, declined or why
        return fallback, declined
