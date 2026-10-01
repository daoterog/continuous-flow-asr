"""Driving the tau refresh: when it happens, which curve it reads, and what it carries."""

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig
from torchmetrics import Metric

from src.flow.schedule import make_tau_inverse, sweep_corruption

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


class TauScheduler:
    """Decides when tau is re-derived, from which curve, and builds the new one.

    `src.flow.schedule` turns a P_e curve into tau; this is the stateful half that picks
    the curve and keeps mu's running accuracy. Separate from the model for the same reason
    `SamplingProbe` is: the model should call it, not contain it.

    A refresh belongs at the start of a training epoch. The metrics it reads still hold the
    previous epoch's counts there — they are logged at epoch end and reset just after this
    runs — and the bins can be re-cut immediately, because nothing has been accumulated
    under the new edges yet.
    """

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """`cfg` is the `schedule` config node; `seed` draws the sweep's tokens."""
        self.cfg = cfg
        self.seed = seed
        # Mu's latest smoothed accuracy, kept here because the reading is taken at the end
        # of a training epoch while the refresh consuming it runs at the next epoch's start.
        self.mu_accuracy: float | None = None
        self._ema = 0.0
        self._ema_steps = 0

    @property
    def curve(self) -> str:
        """Which P_e curve a refresh reads: the corruption sweep, or the denoiser's own."""
        return str(self.cfg.get("refresh_curve_from", "simulated"))

    @property
    def refresh_split(self) -> str:
        """Which split's error the refresh reads."""
        return str(self.cfg.get("refresh_split", "train"))

    def observe_mu(self, metrics: Mapping[str, Metric]) -> None:
        """Fold mu's error on the refresh split into the running accuracy."""
        key = f"{self.refresh_split}_mu_perr"
        if key not in metrics:
            return
        perr = float(metrics[key].overall())
        if math.isnan(perr):
            return
        self.smooth_mu_accuracy(1.0 - perr)

    def smooth_mu_accuracy(self, measured: float) -> float:
        """Bias-corrected EMA of mu's accuracy, so tau tracks its trend and not SGD wobble.

        The result is kept on `mu_accuracy` for the next refresh to pick up.
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

    def is_due(self, model: "FlowMatching") -> bool:
        """Whether tau should be re-derived at the start of this training epoch.

        Epoch 0 never refreshes — the metrics a refresh reads would be empty — so with
        `every = n` the schedule is re-derived at the start of epochs n, 2n, ..., each
        from the epoch that just finished.

        Not gated on mu for the `measured` curve: that one is rebuilt from the denoiser's
        own P_e and needs no mu at all, so a linear run refreshes too. Gating it here is
        what would leave `linear` silently on its initial schedule, which is precisely the
        control the comparison needs.
        """
        every = int(self.cfg.get("refresh_every_n_epochs", 0))
        needs_mu = self.curve == "simulated"
        return (
            every > 0
            and model.sample_t_fn is not None
            and (model.gamma_fn is not None or not needs_mu)
            and not model.trainer.sanity_checking
            and model.current_epoch > 0
            and model.current_epoch % every == 0
        )

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[Callable | None, str | None]:
        """Re-derive tau; `refresh_curve_from` picks the curve it is built from.

        Returns (new tau, None), or (None, a sentence saying why nothing was refreshed) —
        the caller is what has a logger to print it with.
        """
        if datamodule is None:
            return None, "tau refresh skipped: no datamodule to sweep from"
        curve = self.curve

        if curve == "measured":
            key = f"{self.refresh_split}_denoiser_perr"
            if key not in model.metrics:
                return None, f"tau refresh skipped: no {key} metric to read"
            metric = model.metrics[key]
            if not bool(metric.seen().any()):
                return None, f"tau refresh skipped: {key} saw no samples this epoch"
            new_tau = self._measured_tau(metric, model.vocab_size)
        elif curve == "simulated":
            if model.gamma_fn is not None and self.mu_accuracy is None:
                return None, "tau refresh skipped: the simulated curve needs mu's accuracy"
            tokens = datamodule.sample_tokens("train", int(self.cfg.sweep_n), seed=self.seed)
            ts = np.linspace(0.0, 1.0, int(self.cfg.num_bins) + 1).tolist()
            recovery = sweep_corruption(
                tokens.to(model.device),
                model.vocab_size,
                ts,
                alpha_fn=model.alpha_fn,
                beta_fn=model.beta_fn,
                gamma_fn=model.gamma_fn,
                mu_accuracy=0.0 if self.mu_accuracy is None else self.mu_accuracy,
            )
            new_tau = make_tau_inverse(ts, recovery, model.vocab_size)
        else:
            raise ValueError(
                f"schedule.refresh_curve_from must be 'simulated' or 'measured', "
                f"got {curve!r}"
            )

        return new_tau, None

    def _measured_tau(self, metric: Metric, vocab_size: int) -> Callable:
        """Tau from the denoiser's recovery, averaged across refreshes on a fixed t grid."""
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
