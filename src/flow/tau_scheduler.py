"""Driving the tau refresh: when it happens, which curve it reads, and what it carries."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig
from torchmetrics import Metric

from src.flow.schedule import make_tau_inverse, sweep_corruption

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


class TauScheduler:
    """Decides when tau is re-derived, builds it, and holds it until the swap is safe.

    `src.flow.schedule` turns a P_e curve into tau; this is the stateful half that chooses the
    curve, keeps mu's running accuracy and defers the swap. Separate from the model for
    the same reason `SamplingProbe` is: the model should call it, not contain it.

    The new tau is held rather than applied where it is computed. Lightning runs
    validation before `on_train_epoch_end`, so swapping at the point of computation would
    re-cut the bins underneath train counts that had not been logged yet; the model takes
    it at the next `on_train_epoch_start`, by which time both splits have been flushed.
    """

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """`cfg` is the `schedule` config node; `seed` draws the sweep's tokens."""
        self.cfg = cfg
        self.seed = seed
        # The accuracy tau was last rebuilt from. Saved with the run, because rebuilding
        # tau from `schedule.mu_accuracy` on reload gives the *prior*, not the schedule the
        # model actually trained under — they diverge as mu learns.
        self.last_mu_accuracy: float | None = None
        self._pending: tuple[Callable, float | None] | None = None
        self._ema = 0.0
        self._ema_steps = 0

    def smooth_mu_accuracy(self, measured: float) -> float:
        """Bias-corrected EMA of mu's accuracy, so tau tracks its trend and not SGD wobble.

        Call on every validation, not only when a refresh is due: the average should see
        every reading it can, which also makes the momentum's lag count epochs rather than
        refreshes.
        """
        m = float(self.cfg.get("mu_accuracy_momentum", 0.0))
        if not 0.0 <= m < 1.0:
            raise ValueError(f"schedule.mu_accuracy_momentum must be in [0, 1), got {m}")
        if m == 0.0:
            return measured
        self._ema = m * self._ema + (1.0 - m) * measured
        self._ema_steps += 1
        return self._ema / (1.0 - m**self._ema_steps)

    def is_due(self, model: "FlowMatching") -> bool:
        """Whether tau should be re-derived at the end of this validation.

        Not gated on mu for the `measured` curve: that one is rebuilt from the denoiser's
        own P_e and needs no mu at all, so a linear run refreshes too. Gating it here is
        what would leave `linear` silently on its initial schedule, which is precisely the
        control the comparison needs.
        """
        every = int(self.cfg.get("refresh_every_n_epochs", 0))
        needs_mu = str(self.cfg.get("refresh_curve_from", "simulated")) == "simulated"
        return (
            every > 0
            and model.sample_t_fn is not None
            and (model.gamma_fn is not None or not needs_mu)
            and not model.trainer.sanity_checking
            and (model.current_epoch + 1) % every == 0
        )

    def refresh(
        self,
        model: "FlowMatching",
        datamodule: "AudioFeatureDataModule | None",
        mu_accuracy: float | None,
    ) -> str | None:
        """Re-derive tau and hold it as pending; `refresh_curve_from` picks the curve.

        Returns None once the new tau is held, or a sentence saying why nothing was
        refreshed — the caller is what has a logger to print it with.
        """
        if datamodule is None:
            return "tau refresh skipped: no datamodule to sweep from"
        curve = str(self.cfg.get("refresh_curve_from", "simulated"))

        if curve == "measured":
            key = f"{self.cfg.get('refresh_split', 'train')}_denoiser_perr"
            if key not in model.metrics:
                return f"tau refresh skipped: no {key} metric to read"
            new_tau = self._measured_tau(model.metrics[key], model.vocab_size)
        elif curve == "simulated":
            if model.gamma_fn is not None and mu_accuracy is None:
                return "tau refresh skipped: the simulated curve needs mu's accuracy"
            tokens = datamodule.sample_tokens("train", int(self.cfg.sweep_n), seed=self.seed)
            ts = np.linspace(0.0, 1.0, int(self.cfg.num_bins) + 1).tolist()
            recovery = sweep_corruption(
                tokens.to(model.device),
                model.vocab_size,
                ts,
                alpha_fn=model.alpha_fn,
                beta_fn=model.beta_fn,
                gamma_fn=model.gamma_fn,
                mu_accuracy=0.0 if mu_accuracy is None else mu_accuracy,
            )
            new_tau = make_tau_inverse(ts, recovery, model.vocab_size)
        else:
            raise ValueError(
                f"schedule.refresh_curve_from must be 'simulated' or 'measured', "
                f"got {curve!r}"
            )

        if mu_accuracy is not None:
            self.last_mu_accuracy = mu_accuracy
        self._pending = (new_tau, mu_accuracy)
        return None

    def take_pending(self) -> tuple[Callable, float | None] | None:
        """The tau waiting to be applied, cleared as it is handed over."""
        pending, self._pending = self._pending, None
        return pending

    def _measured_tau(self, metric: Metric, vocab_size: int) -> Callable:
        """Tau from the denoiser's recovery, averaged across refreshes on a fixed t grid."""
        seen = metric.seen().detach().cpu().numpy()
        ts = metric.bin_centers().numpy()[seen]
        rec = metric.recovery().detach().cpu().numpy()[seen]
        if rec.size == 0 or rec.max() < 0.05:
            raise ValueError(
                f"the model recovers at most {rec.max() if rec.size else 0:.3f} of tokens "
                f"in any bin; tau would collapse onto t=1. Train further first."
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
