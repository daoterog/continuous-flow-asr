"""Reading a components directory back into usable models."""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.flow.schedule import TimeReparametrization
from src.models import build_audio_decoder, build_denoiser, build_two_time_denoiser

from .layout import BUILDERS, MANIFEST, TAU, VOCAB_MAP


def load_manifest(run_dir: Path) -> DictConfig:
    """Read a run's manifest. `run_dir` is the directory holding manifest.yaml."""
    path = Path(run_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"no {MANIFEST} under {run_dir}; pass the components/ directory a run wrote"
        )
    return OmegaConf.load(path)


def load_tau(run_dir: Path, variant: str = "final") -> TimeReparametrization | None:
    """The time reparametrization saved with `<name>_<variant>.pth`, or None if none was."""
    path = Path(run_dir) / f"{TAU}_{variant}.npz"
    return TimeReparametrization.load(path) if path.exists() else None


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
) -> tuple[DictConfig, dict[str, torch.nn.Module], TimeReparametrization | None]:
    """Rebuild everything a run saved that another model might want from it.

    Returns (manifest, {name: module}, reparametrization). The modules are in eval mode and
    carry no optimizer state — this is the path for distillation and inference, not for
    resuming.

    The curve comes back alongside the weights because it is as much a part of the
    checkpoint as they are: the weights were trained against it, and a refreshed one cannot
    be rebuilt from the config the run started with. Loading them together is what stops a
    caller pairing one run's weights with another run's grid. None when the run used the
    identity, which any caller rebuilds for free.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    built: dict[str, torch.nn.Module] = {}

    model_cfg = OmegaConf.create(manifest.model)
    for name in BUILDERS:
        path = run_dir / f"{name}_{variant}.pth"
        if not path.exists():
            continue
        state = torch.load(path, map_location=map_location, weights_only=True)
        if name == "mu_nn":
            if model_cfg.get("audio_decoder") is None:
                raise ValueError(
                    f"{path.name} exists but the manifest has no model.audio_decoder"
                )
            # A run with flow.learned_noise saved mu with its noise head; the weights say so.
            model = build_audio_decoder(
                model_cfg.audio_decoder, manifest.vocab_size, manifest.canvas_len,
                noise_head="noise_head.weight" in state,
            )
        elif name == "two_time_denoiser":
            # The same `model.denoiser` node as the one-time model: a two-time denoiser is
            # that geometry plus one extra time embedder, which is what lets a teacher's
            # weights transfer into it.
            model = build_two_time_denoiser(model_cfg.denoiser, manifest.vocab_size)
        else:
            model = build_denoiser(model_cfg.denoiser, manifest.vocab_size)
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
    return manifest, built, load_tau(run_dir, variant)
