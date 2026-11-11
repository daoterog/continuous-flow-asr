"""The flow denoiser: DRAX's Transformer, configured from our YAML.

DRAX's flow decoder is used unmodified: DDiTBlock is self-attn (RoPE) -> cross-attn to
audio -> MLP, each sub-layer shift/scale/gated by adaLN off the time embedding. RoPE is
applied only inside self-attention — text positions and audio frames are different
coordinate systems, so rotating the cross-attention queries against audio keys would be
meaningless. Licence is CC-BY-NC (Meta / aiOla): research use, keep the headers.

The Transformer takes a 3D x_t directly — it runs ``x_t @ vocab_embed.weight`` instead of
an embedding lookup — so the continuous simplex field never has to be discretised.
"""

from src.constants import WHISPER_DIM

from omegaconf import DictConfig, OmegaConf

from .drax.transformer import Transformer

# Keys the Transformer reads off its config. Listed here so a typo in the YAML fails with
# a useful message rather than an AttributeError six frames down.
REQUIRED_KEYS = ("hidden_size", "cond_dim", "n_heads", "n_blocks", "dropout")


def build_denoiser(cfg: DictConfig | dict, vocab_size: int) -> Transformer:
    """Build the denoiser from the `model.denoiser` node of the config.

    `whisper_dim` defaults to the large-v3 encoder width; it is what
    ``Transformer.audio_proj`` projects the cached latents from, so it has to match the
    checkpoint the features were precomputed with, not the model being trained.

    masked=False: this is the continuous flow, so there is no extra mask token in the
    vocabulary. DRAX zero-inits the final layer, which is why an untrained denoiser's
    cross-entropy is exactly ln(V).
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"model.denoiser is missing {missing}; it needs {list(REQUIRED_KEYS)} "
            f"(plus an optional whisper_dim, default {WHISPER_DIM})"
        )

    cfg = OmegaConf.merge(OmegaConf.create({"whisper_dim": WHISPER_DIM}), cfg)
    return Transformer(vocab_size=vocab_size, masked=False, config=cfg)
