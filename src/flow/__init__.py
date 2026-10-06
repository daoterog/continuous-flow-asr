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


def build_tau_scheduler(cfg: DictConfig, seed: int = 0) -> TauScheduler:
    """The tau strategy `cfg.refresh_curve_from` names; `cfg` is the `schedule` node.

    Raises on an unknown name here rather than at the first refresh, which with
    `refresh_every_n_epochs: 5` would be several epochs into a job.
    """
    curve = str(cfg.get("refresh_curve_from", "simulated"))
    if curve not in TAU_STRATEGIES:
        raise ValueError(
            f"schedule.refresh_curve_from must be one of "
            f"{sorted(TAU_STRATEGIES)}, got {curve!r}"
        )
    return TAU_STRATEGIES[curve](cfg, seed=seed)
