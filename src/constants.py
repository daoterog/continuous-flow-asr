"""Corpus layout constants, plus the import-time path and cache bootstrap.

Two of these side effects have to happen *before* other imports, which is why this
module is imported for its effects as well as its names:

  * ``normalizer`` lives in the ``open_asr_leaderboard`` submodule, which is not on the
    path until we put it there;
  * ``HF_HOME`` is resolved by huggingface at *import* time, so setting it after
    ``import transformers`` is too late and the weights land in ``~/.cache``.

So any module that imports ``transformers`` or ``normalizer`` starts with::

    import src.constants  # noqa: F401

The constants below are duplicated from ``scripts/data/download_librispeech.py`` and
``scripts/data/precompute_whisper.py`` on purpose: training must not drag the download and
precompute entry points (and their argparse setup) into every import. They are a fixed
property of the corpus, so they do not drift — but if one of those scripts changes,
change it here too.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = REPO_ROOT / "open_asr_leaderboard"

assert (LEADERBOARD / "normalizer" / "__init__.py").is_file(), (
    f"{LEADERBOARD} is empty — run: git submodule update --init --recursive"
)

for _p in (REPO_ROOT, LEADERBOARD):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REPO_ID = "openslr/librispeech_asr"
DEFAULT_MODEL = "openai/whisper-large-v3"

# `configs` x `splits` is a cross product; pairs that do not exist are skipped rather
# than rejected, so one splits list works for both configs. `all` is deliberately absent:
# it is the same audio as clean + other under merged split names.
CONFIGS = ["clean", "other"]
SPLITS = ["train.100", "train.360", "train.500", "validation", "test"]

SNELLIUS_DATA_ROOT = Path("/scratch-shared/cont-flow-asr/data")
LOCAL_DATA_ROOT = REPO_ROOT / "data"

# large-v3 encoder width, projected by Transformer.audio_proj.
WHISPER_DIM = 1280
# Whisper emits 1500 encoder frames for its fixed 30 s window, one per 20 ms.
WHISPER_FRAMES = 1500

# <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
PREFIX_LEN = 4

# Respect an HF_HOME already set by the batch scripts; otherwise point at data/, where
# the weights actually live.
os.environ.setdefault("HF_HOME", str(LOCAL_DATA_ROOT))


def data_root_for(env: str = "local", data_root: Path | None = None) -> Path:
    """Pick the data root, mirroring the scripts' ``--env`` / ``--data-root`` pair."""
    if data_root is not None:
        return Path(data_root)
    if env == "snellius":
        return SNELLIUS_DATA_ROOT
    if env == "local":
        return LOCAL_DATA_ROOT
    raise ValueError(f"unknown env {env!r}; expected 'local' or 'snellius'")


def feature_dir(data_root: Path, model: str, config: str, split: str) -> Path:
    """Where ``precompute_whisper.py`` wrote the encoder latents for one (config, split).

    Must match ``precompute_whisper.output_paths``, which strips the org off the model
    id: ``openai/whisper-large-v3`` caches under ``features/whisper-large-v3/``.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIGS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return Path(data_root) / "features" / model.split("/")[-1] / config / split
