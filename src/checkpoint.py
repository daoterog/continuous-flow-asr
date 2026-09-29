"""Save a run as loadable parts rather than one opaque Lightning checkpoint.

A `.ckpt` is a pickle of the whole LightningModule: it only loads back into the same class
with the same constructor arguments, and `torch.load` on it has to run arbitrary code. That
is the wrong shape for distillation, where the student wants the denoiser's weights and
nothing else.

So a run writes a directory instead:

    runs/<run_id>/components/
        manifest.yaml     everything build_denoiser / build_audio_decoder need
        vocab_map.pt      compact id -> Whisper id, without which the tokens mean nothing
        denoiser_best.pth , denoiser_final.pth
        mu_nn_best.pth    , mu_nn_final.pth
        training_state.pt optimizer + scheduler + epoch, for resuming a cut-short run

Every `.pth` is a plain state_dict loaded with `weights_only=True`. The manifest is YAML
so it can be read without torch at all.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf

from src.models import build_audio_decoder, build_denoiser

if TYPE_CHECKING:
    from src.dataset import VocabMap

# The parts a student model might want, in the order they are reported.
COMPONENTS = ("denoiser", "mu_nn")
MANIFEST = "manifest.yaml"
VOCAB_MAP = "vocab_map.pt"
TRAINING_STATE = "training_state.pt"


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """The module `torch.compile` wrapped, or the module itself.

    An OptimizedModule's state_dict prefixes every key with `_orig_mod.`, which will not
    load into a plain build_denoiser(...). Saving the inner module keeps the file loadable
    whether or not the run that produced it used compile.
    """
    return getattr(module, "_orig_mod", module)


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
    """Write each model's weights under `<name>_<variant>.pth`. Returns what was written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in COMPONENTS:
        model = getattr(module, name, None)
        if model is None:  # mu_nn is absent on the linear variant
            continue
        path = out_dir / f"{name}_{variant}.pth"
        torch.save(_unwrap(model).state_dict(), path)
        written.append(path)
    return written


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


def strip_compile_prefix(state: dict) -> dict:
    """Drop the `_orig_mod.` torch.compile adds, so weights load into a plain module."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def remap_legacy_keys(state: dict) -> dict:
    """Rename audio-decoder keys written before dropout was inserted into its FF stack.

    `AudioDecoder.ff` is an nn.Sequential, so adding `nn.Dropout` between the activation
    and the output projection renumbered the projection from 2 to 3 — the dropout itself
    holds no parameters, but everything after it shifted. Checkpoints written before that
    change carry `ff.2.weight`, which the current module has no slot for.

    A `ff.2.weight` is the discriminator: in the current layout index 2 is the dropout and
    never has one. Runs up to `trim-sound-9` need this; later ones pass through untouched.
    """
    if not any(k.endswith(".ff.2.weight") for k in state):
        return state
    return {k.replace(".ff.2.", ".ff.3."): v for k, v in state.items()}


def load_state_into(model: torch.nn.Module, state: dict, name: str) -> None:
    """Load one component's weights, undoing compile prefixes and legacy naming first."""
    prefix = f"{name}."
    own = {
        k[len(prefix):]: v
        for k, v in strip_compile_prefix(state).items()
        if k.startswith(prefix)
    }
    if not own:
        raise KeyError(f"no `{prefix}*` keys in the state dict; found {sorted(state)[:4]}")
    _unwrap(model).load_state_dict(remap_legacy_keys(own), strict=True)


def run_dir(run_id: str, save_dir: str | Path = "runs") -> Path:
    """`<save_dir>/<run_id>/components` — where a run of that wandb id wrote its parts."""
    path = Path(save_dir) / str(run_id) / "components"
    if not path.exists():
        found = sorted(p.name for p in Path(save_dir).glob("*/components/..")) or "none"
        raise FileNotFoundError(f"no components for run {run_id!r} under {save_dir} ({found})")
    return path


def load_manifest(run_dir: Path) -> DictConfig:
    """Read a run's manifest. `run_dir` is the directory holding manifest.yaml."""
    path = Path(run_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"no {MANIFEST} under {run_dir}; pass the components/ directory a run wrote"
        )
    return OmegaConf.load(path)


def load_vocab_map(run_dir: Path, tokenizer_size: int) -> torch.Tensor | None:
    """`to_full` for a compact-vocabulary run, or None if the run used the full vocab."""
    path = Path(run_dir) / VOCAB_MAP
    if not path.exists():
        return None
    to_full = torch.load(path, weights_only=True)["to_full"]
    if int(to_full.max()) >= tokenizer_size:
        raise ValueError(
            f"{VOCAB_MAP} maps to id {int(to_full.max())} but the tokenizer has "
            f"{tokenizer_size} — it was saved against a different checkpoint"
        )
    return to_full


def load_components(
    run_dir: Path, variant: str = "final", map_location: str | torch.device = "cpu"
) -> tuple[DictConfig, dict[str, torch.nn.Module]]:
    """Rebuild the models from a manifest and load `<name>_<variant>.pth` into them.

    Returns (manifest, {name: module}). The modules are in eval mode and carry no
    optimizer state — this is the path for distillation and inference, not for resuming.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    built: dict[str, torch.nn.Module] = {}

    model_cfg = OmegaConf.create(manifest.model)
    for name in COMPONENTS:
        path = run_dir / f"{name}_{variant}.pth"
        if not path.exists():
            continue
        if name == "denoiser":
            model = build_denoiser(model_cfg.denoiser, manifest.vocab_size)
        else:
            if model_cfg.get("audio_decoder") is None:
                raise ValueError(
                    f"{path.name} exists but the manifest has no model.audio_decoder"
                )
            model = build_audio_decoder(
                model_cfg.audio_decoder, manifest.vocab_size, manifest.canvas_len
            )
        state = torch.load(path, map_location=map_location, weights_only=True)
        # strict: a silently partial load is the failure this whole format exists to
        # avoid, and it raises with the offending key names.
        model.load_state_dict(state, strict=True)
        model.eval()
        built[name] = model

    if not built:
        raise FileNotFoundError(
            f"no component weights for variant '{variant}' under {run_dir}; "
            f"found {sorted(p.name for p in run_dir.glob('*.pth'))}"
        )
    return manifest, built


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
        """Write `_final` and, unless told not to, the state needed to resume."""
        save_components(module, self.out_dir, "final")
        if self.save_training_state:
            save_training_state(trainer, self.out_dir)

    def teardown(
        self, trainer: pl.Trainer, module: pl.LightningModule, stage: str
    ) -> None:
        """Same, on the path SLURM takes: a signalled run reaches teardown, not on_fit_end."""
        if stage == "fit":
            self.on_fit_end(trainer, module)


def copy_into(run_dir: Path, dest: Path) -> list[Path]:
    """Copy a components directory somewhere else, e.g. a wandb run dir before upload."""
    run_dir, dest = Path(run_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, dest / path.name)
            copied.append(dest / path.name)
    return copied
