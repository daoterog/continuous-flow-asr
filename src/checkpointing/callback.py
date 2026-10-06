"""The Lightning callback that writes the parts out as training goes."""

from pathlib import Path
from typing import TYPE_CHECKING

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from .save import save_components, save_manifest, save_training_state

if TYPE_CHECKING:
    from src.data import VocabMap


class ComponentCheckpoint(pl.Callback):
    """Write the run's parts to `<save_dir>/<run_id>/components` as training goes.

    Three moments: the manifest once at the start, so a crashed run is still rebuildable;
    `_best` whenever the monitored metric improves; `_final` plus the training state when
    fit ends, including when SLURM cuts it short — Lightning runs teardown on the signal.
    """

    def __init__(
        self,
        out_dir: Path,
        manifest: DictConfig,
        vocab_map: "VocabMap | None" = None,
        monitor: str = "total_loss/val",
        mode: str = "min",
        save_training_state: bool = True,
    ) -> None:
        """Track `monitor` for the `_best` variant; `mode` says which direction wins."""
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.out_dir = Path(out_dir)
        self.manifest = manifest
        self.vocab_map = vocab_map
        self.monitor = monitor
        self.mode = mode
        self.save_training_state = save_training_state
        self.best = float("inf") if mode == "min" else float("-inf")

    def on_fit_start(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        """Write the manifest before any weights exist, so a crash still leaves a rebuild."""
        save_manifest(self.out_dir, self.manifest, self.vocab_map)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, module: pl.LightningModule
    ) -> None:
        """Refresh `_best` when the monitored metric improves."""
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        value = float(value)
        better = value < self.best if self.mode == "min" else value > self.best
        if not better:
            return
        self.best = value
        save_components(module, self.out_dir, "best")

    def on_fit_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        """Write `_final`, the settled schedule, and the state needed to resume."""
        # tau is rebuilt from schedule.mu_accuracy on load, which is the *prior* — the
        # value assumed before training. A refreshing run ends somewhere else entirely
        # (0.43 -> 0.20 on the full vocabulary), so without this the model is sampled on a
        # different grid than it trained on, silently.
        #
        # Only for a run that actually refreshed. One that did not trained on the prior all
        # the way through, and recording where mu happened to settle would send sample_eval
        # off to rebuild a schedule the model never saw.
        scheduler = getattr(module, "tau_scheduler", None)
        settled = getattr(scheduler, "mu_accuracy", None)
        if settled is not None and int(scheduler.cfg.get("refresh_every_n_epochs", 0)) > 0:
            self.manifest = OmegaConf.merge(
                self.manifest, OmegaConf.create({"mu_accuracy_final": float(settled)})
            )
            save_manifest(self.out_dir, self.manifest, self.vocab_map)
        save_components(module, self.out_dir, "final")
        if self.save_training_state:
            save_training_state(trainer, self.out_dir)

    def teardown(
        self, trainer: pl.Trainer, module: pl.LightningModule, stage: str
    ) -> None:
        """Same, on the path SLURM takes: a signalled run reaches teardown, not on_fit_end."""
        if stage == "fit":
            self.on_fit_end(trainer, module)
