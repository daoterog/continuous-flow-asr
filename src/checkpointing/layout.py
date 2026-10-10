"""Where a run's parts live on disk, and the names they are written under."""

import shutil
from pathlib import Path

# What an ordinary flow-matching run saves, in the order they are reported. A module may
# override this through its `components` property — a distillation run trains a
# `two_time_denoiser` and holds `denoiser`/`mu_nn` as a frozen teacher, so saving this tuple
# off it would write the teacher straight back out and drop the only weights that moved.
COMPONENTS = ("denoiser", "mu_nn")
# Every name any module may ask to have saved, and how to rebuild each one. Keyed here
# rather than branched on inside `load_components` so adding a model is one entry.
BUILDERS = ("denoiser", "mu_nn", "two_time_denoiser")
MANIFEST = "manifest.yaml"
VOCAB_MAP = "vocab_map.pt"
TRAINING_STATE = "training_state.pt"
# tau^-1 as live when a variant was written, `tau_<variant>.npz`.
TAU = "tau"

def run_dir(run_id: str, save_dir: str | Path = "runs") -> Path:
    """`<save_dir>/<run_id>/components` — where a run of that wandb id wrote its parts."""
    path = Path(save_dir) / str(run_id) / "components"
    if not path.exists():
        found = sorted(p.name for p in Path(save_dir).glob("*/components/..")) or "none"
        raise FileNotFoundError(f"no components for run {run_id!r} under {save_dir} ({found})")
    return path

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
