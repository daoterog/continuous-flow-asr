"""Dataset, compact vocabulary and the DataModule over the precomputed Whisper latents.

Splits are named the way the download and precompute scripts name them: a role (train,
val, test) is a set of `configs` x `splits` pairs, each resolving to one feature
directory under `data/features/<model>/<config>/<split>/`. That keeps
``--train-configs clean other --train-splits train.100 train.360`` meaning the same thing
here as it does for ``scripts/data/precompute_whisper.py``.
"""

from .collate import collate, make_loader
from .datamodule import ROLES, AudioFeatureDataModule
from .features import AudioFeatureDataset
from .spec import expand_spec, resolve_role
from .vocab import VocabMap

__all__ = [
    "ROLES",
    "AudioFeatureDataModule",
    "AudioFeatureDataset",
    "VocabMap",
    "collate",
    "expand_spec",
    "make_loader",
    "resolve_role",
]
