"""The probability path: its coefficients, its time reparameterization, and sampling.

Each pure module is paired with the stateful driver that runs it against a model —
``schedule`` with the ``tau_*`` strategies, ``sampler`` with ``sampling_probe``. The
drivers know about the LightningModule; the modules they drive know only tensors.

Only ``build_tau_scheduler`` is exported, because picking a tau strategy is the one thing
that has to know about more than one of these modules at once. Everything else is imported
from its own module, so this package's import order stays irrelevant.
"""

from omegaconf import DictConfig

from .tau_base import TauScheduler
from .tau_measured import MeasuredTau
from .tau_simulated import SimulatedTau
from .tau_warmup import WarmupTau

__all__ = [
    "MeasuredTau",
    "SimulatedTau",
    "TauScheduler",
    "WarmupTau",
    "build_tau_scheduler",
]

# `schedule.refresh_curve_from` selects one of these.
TAU_STRATEGIES: dict[str, type[TauScheduler]] = {
    "simulated": SimulatedTau,
    "measured": MeasuredTau,
    "warmup": WarmupTau,
}


def build_tau_scheduler(cfg: DictConfig, seed: int = 0, val_every: int = 1) -> TauScheduler:
    """The tau strategy `cfg.refresh_curve_from` names; `cfg` is the `schedule` node.

    Raises on an unknown name here rather than at the first refresh, which with
    `refresh_every_n_epochs: 5` would be several epochs into a job.

    A refresh that reads the val split must land on a validation boundary: every
    `refresh_every_n_epochs` a multiple of `val_every` (`train.check_val_every_n_epoch`),
    so the curve is re-read right after a validation pass and never between two of them.
    """
    curve = str(cfg.get("refresh_curve_from", "simulated"))
    if curve not in TAU_STRATEGIES:
        raise ValueError(
            f"schedule.refresh_curve_from must be one of "
            f"{sorted(TAU_STRATEGIES)}, got {curve!r}"
        )
    every = int(cfg.get("refresh_every_n_epochs", 0))
    if str(cfg.get("refresh_split", "train")) == "val" and every % max(val_every, 1):
        raise ValueError(
            f"schedule.refresh_split=val refreshes every {every} epochs, which is not a "
            f"multiple of train.check_val_every_n_epoch={val_every}; set "
            f"refresh_every_n_epochs to a multiple of it"
        )
    return TAU_STRATEGIES[curve](cfg, seed=seed)
