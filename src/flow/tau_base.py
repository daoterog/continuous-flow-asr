"""What every tau refresh shares: its cadence, and the contract the model calls through."""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from omegaconf import DictConfig
from torchmetrics import Metric

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


class TauScheduler:
    """Decides when tau is re-derived; a subclass decides what it is re-derived from.

    `src.flow.schedule` turns a P_e curve into tau, and which curve that is amounts to a
    whole strategy rather than a flag: `SimulatedTau` replays the corruption on real
    tokens, `MeasuredTau` reads the denoiser's own error. They take different inputs, need
    different things to exist, and fail for different reasons, so they are separate
    objects — `src.flow.build_tau_scheduler` picks one from the config.

    A refresh belongs at the start of a training epoch. The metrics it reads still hold the
    previous epoch's counts there — they are logged at epoch end and reset just after this
    runs — and the bins can be re-cut immediately, because nothing has been accumulated
    under the new edges yet.

    Separate from the model for the same reason `SamplingProbe` is: the model should call
    it, not contain it.
    """

    # Whether the strategy is pointless without a mu reading the audio.
    needs_mu: bool = False
    # Mu's smoothed accuracy, for whichever strategy tracks one. Read by the reporting
    # mixin and by the checkpoint manifest, so it is part of the base contract even where
    # it stays None.
    mu_accuracy: float | None = None

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """`cfg` is the `schedule` config node; `seed` draws whatever sampling is needed."""
        self.cfg = cfg
        self.seed = seed

    @property
    def refresh_split(self) -> str:
        """Which split's error the refresh reads."""
        return str(self.cfg.get("refresh_split", "train"))

    def observe_mu(self, metrics: Mapping[str, Metric]) -> None:
        """Take mu's reading for the epoch just finished.

        A no-op unless the strategy has a use for it, so the caller can run it every epoch
        without asking which strategy it holds.
        """

    def is_due(self, model: "FlowMatching") -> bool:
        """Whether tau should be re-derived at the start of this training epoch.

        Epoch 0 never refreshes — the metrics a refresh reads would be empty — so with
        `every = n` the schedule is re-derived at the start of epochs n, 2n, ..., each
        from the epoch that just finished.

        A strategy that does not need mu refreshes even on a run without one, which is what
        keeps `linear` from sitting silently on its initial schedule while the variants it
        is the control for move.
        """
        every = int(self.cfg.get("refresh_every_n_epochs", 0))
        return (
            every > 0
            and model.sample_t_fn is not None
            and (model.gamma_fn is not None or not self.needs_mu)
            and not model.trainer.sanity_checking
            and model.current_epoch > 0
            and model.current_epoch % every == 0
        )

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[Callable | None, str | None]:
        """Re-derive tau from this strategy's curve.

        Returns (new tau, None), or (None, a sentence saying why nothing was refreshed) —
        the caller is what has a logger to print it with.
        """
        raise NotImplementedError
