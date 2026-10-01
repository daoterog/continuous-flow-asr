"""Where a run's parts live on disk, and the names they are written under."""

import shutil
from pathlib import Path

# The parts a student model might want, in the order they are reported.
COMPONENTS = ("denoiser", "mu_nn")
MANIFEST = "manifest.yaml"
VOCAB_MAP = "vocab_map.pt"
TRAINING_STATE = "training_state.pt"

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
