"""Reading a components directory back into usable models."""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.models import build_audio_decoder, build_denoiser

from .layout import COMPONENTS, MANIFEST, VOCAB_MAP


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
