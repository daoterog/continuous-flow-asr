"""Reconciling a saved state_dict with the module layout in front of it."""

import torch


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """The module `torch.compile` wrapped, or the module itself.

    An OptimizedModule's state_dict prefixes every key with `_orig_mod.`, which will not
    load into a plain build_denoiser(...). Saving the inner module keeps the file loadable
    whether or not the run that produced it used compile.
    """
    return getattr(module, "_orig_mod", module)

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
