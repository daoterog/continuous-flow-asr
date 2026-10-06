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
        tau_best.npz      , tau_final.npz   the tau^-1 live when each variant was written
        training_state.pt optimizer + scheduler + epoch, for resuming a cut-short run

Every `.pth` is a plain state_dict loaded with `weights_only=True`. The manifest is YAML
so it can be read without torch at all.
"""

from .callback import ComponentCheckpoint
from .layout import COMPONENTS, MANIFEST, TAU, TRAINING_STATE, VOCAB_MAP, copy_into, run_dir
from .load import load_components, load_manifest, load_tau, load_vocab_map
from .save import manifest_for, save_components, save_manifest, save_training_state
from .state_dict import load_state_into, remap_legacy_keys, strip_compile_prefix

__all__ = [
    "COMPONENTS",
    "MANIFEST",
    "TAU",
    "TRAINING_STATE",
    "VOCAB_MAP",
    "ComponentCheckpoint",
    "copy_into",
    "load_components",
    "load_manifest",
    "load_tau",
    "load_state_into",
    "load_vocab_map",
    "manifest_for",
    "remap_legacy_keys",
    "run_dir",
    "save_components",
    "save_manifest",
    "save_training_state",
    "strip_compile_prefix",
]
