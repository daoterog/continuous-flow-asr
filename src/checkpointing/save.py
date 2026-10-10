"""Writing a run out: the rebuild manifest, the weights, and the state to resume."""

from pathlib import Path
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf

from .layout import COMPONENTS, MANIFEST, TAU, TRAINING_STATE, VOCAB_MAP
from .state_dict import _unwrap

if TYPE_CHECKING:
    from src.data import VocabMap


def manifest_for(
    cfg: DictConfig,
    vocab_size: int,
    pad_id: int,
    audio_dim: int,
    canvas_len: int,
) -> DictConfig:
    """Everything needed to rebuild the models, and nothing about how they were trained.

    `flow` rides along because the sampler needs the same alpha/beta/gamma the model was
    trained under; a denoiser is meaningless without the probability path it denoises.
    """
    return OmegaConf.create(
        {
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "flow": OmegaConf.to_container(cfg.flow, resolve=True),
            "vocab_size": int(vocab_size),
            "pad_id": int(pad_id),
            "audio_dim": int(audio_dim),
            "canvas_len": int(canvas_len),
            "tokenizer": str(cfg.data.model),
            "compact_vocab": bool(cfg.data.compact_vocab),
            "variant": cfg.get("variant"),
            "size": cfg.get("size"),
            "mu_size": cfg.get("mu_size"),
            # The whole resolved config, so a components directory is enough to rebuild
            # the data pipeline too — not just the models. The keys above stay spelled out
            # rather than being read back out of this, because they are the contract and
            # this is a record.
            "run_config": OmegaConf.to_container(cfg, resolve=True),
        }
    )


def save_components(
    module: pl.LightningModule, out_dir: Path, variant: str = "final"
) -> list[Path]:
    """Write each model's weights under `<name>_<variant>.pth`. Returns what was written.

    Which models those are is the module's to say, through its `components` property. A
    distillation run holds `denoiser` and `mu_nn` as a frozen teacher and trains a
    `two_time_denoiser`, so saving a fixed tuple off it would write the teacher back out
    unchanged and drop the only weights that moved — with no error, since the files would
    still appear.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in getattr(module, "components", COMPONENTS):
        model = getattr(module, name, None)
        if model is None:  # mu_nn is absent on the linear variant
            continue
        path = out_dir / f"{name}_{variant}.pth"
        torch.save(_unwrap(model).state_dict(), path)
        written.append(path)
    tau_path = save_tau(module, out_dir, variant)
    if tau_path is not None:
        written.append(tau_path)
    return written


def save_tau(module: pl.LightningModule, out_dir: Path, variant: str) -> Path | None:
    """Write the time reparametrization live at this moment as `tau_<variant>.npz`.

    The weights were trained against that curve: under `flow.time_input: uniform` it is
    what pairs each time input u with its flow time t, and in any case it is the grid they
    should be sampled on. A refreshed curve, a measured one above all, cannot be rebuilt
    from the config, so it travels with the weights it belongs to. Nothing is written for
    the identity, which any loader rebuilds for free.
    """
    reparam = getattr(module, "time_reparam", None)
    if reparam is None or reparam.is_identity:
        return None
    path = Path(out_dir) / f"{TAU}_{variant}.npz"
    reparam.save(path)
    return path


def save_manifest(
    out_dir: Path, manifest: DictConfig, vocab_map: "VocabMap | None" = None
) -> None:
    """Write the rebuild manifest, and the vocabulary map when the run used one."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(manifest, out_dir / MANIFEST)
    if vocab_map is not None:
        # to_compact is a (len(tokenizer),) inverse of to_full, so only to_full is stored
        # and the inverse is rebuilt on load — half the file and one source of truth.
        torch.save({"to_full": vocab_map.to_full.cpu()}, out_dir / VOCAB_MAP)


def save_training_state(trainer: pl.Trainer, out_dir: Path) -> Path:
    """Optimizer, scheduler and position in the run — everything weights alone lose.

    Kept apart from the component files because it is an order of magnitude larger and is
    useless to a student model. Overwritten each time, since only the newest can resume.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TRAINING_STATE
    torch.save(
        {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "optimizers": [o.state_dict() for o in trainer.optimizers],
            "schedulers": [
                s.scheduler.state_dict() for s in trainer.lr_scheduler_configs
            ],
        },
        path,
    )
    return path
