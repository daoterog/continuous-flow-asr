"""Resolving a role's `configs` x `splits` pairs to feature directories on disk."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from torch.utils.data import ConcatDataset

from src.constants import CONFIGS, SPLITS, feature_dir

from .features import AudioFeatureDataset


def expand_spec(configs: Iterable[str], splits: Iterable[str]) -> list[tuple[str, str]]:
    """Cross `configs` with `splits`, as the download and precompute scripts do.

    Validated against the whitelists so a typo fails here rather than as an empty role
    after the feature directories are probed.
    """
    configs, splits = list(configs), list(splits)
    for name, got, allowed in (("config", configs, CONFIGS), ("split", splits, SPLITS)):
        bad = [x for x in got if x not in allowed]
        if bad:
            raise ValueError(f"unknown {name}(s) {bad}; expected from {allowed}")
    return [(c, s) for c in configs for s in splits]


def resolve_role(
    data_root: Path, model: str, configs: Iterable[str], splits: Iterable[str], role: str
) -> list[Path]:
    """Feature directories for one role, skipping (config, split) pairs that do not exist.

    Skipping rather than rejecting matches the scripts: LibriSpeech has train.clean.100,
    train.clean.360 and train.other.500, so `--configs clean other --splits train.100
    train.360 train.500` is the natural way to ask for all three, and two of its six pairs
    are not real. A role that resolves to nothing is still an error.
    """
    pairs = expand_spec(configs, splits)
    found = [
        d
        for c, s in pairs
        if (d := feature_dir(data_root, model, c, s)).is_dir() and any(d.glob("*.json"))
    ]
    if not found:
        wanted = ", ".join(f"{c}/{s}" for c, s in pairs)
        raise FileNotFoundError(
            f"no feature directories for role '{role}' ({wanted}) under "
            f"{data_root / 'features' / model.split('/')[-1]}.\nPrecompute them with:\n"
            f"  CONFIGS=\"{' '.join(sorted({c for c, _ in pairs}))}\" "
            f"SPLITS=\"{' '.join(sorted({s for _, s in pairs}))}\" NO_TRIM=1 "
            f"./jobs/submit/whisper.sh"
        )
    return found
