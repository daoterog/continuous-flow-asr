"""mu_psi: the audio-only token decoder that gives the interpolant its third term."""

from src.constants import WHISPER_DIM

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

# Keys the decoder reads off its config, mirroring denoiser.REQUIRED_KEYS. d_ff, dropout
# and audio_dim are optional and defaulted below.
REQUIRED_KEYS = ("hidden", "n_heads", "n_blocks")


class AudioDecoder(nn.Module):
    """mu_psi: token distributions from audio alone, over the fixed canvas.

    The audio frames and the canvas are different lengths with no alignment between them,
    so the canvas is a set of learned queries that cross-attend into the latents rather
    than a position-by-position map. No time conditioning: mu depends only on audio, so
    there is no adaLN here.
    """

    def __init__(
        self,
        audio_dim: int,
        vocab_size: int,
        canvas_len: int,
        hidden: int,
        n_heads: int,
        n_blocks: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Build the learned canvas queries and the self/cross/FF stack over them.

        d_ff defaults to 4 * hidden, the usual transformer ratio.
        """
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * hidden
        self.queries = nn.Parameter(torch.randn(1, canvas_len, hidden) * 0.02)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "n1": nn.LayerNorm(hidden),
                    "sa": nn.MultiheadAttention(
                        hidden, n_heads, dropout=dropout, batch_first=True
                    ),
                    "n2": nn.LayerNorm(hidden),
                    "xa": nn.MultiheadAttention(
                        hidden, n_heads, dropout=dropout, batch_first=True
                    ),
                    "n3": nn.LayerNorm(hidden),
                    "ff": nn.Sequential(
                        nn.Linear(hidden, d_ff),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_ff, hidden),
                    ),
                    # MultiheadAttention's own `dropout` only masks attention weights.
                    # These cover what the standard block also drops: each sub-layer's
                    # output, on the way into the residual. Without them `dropout` leaves
                    # the feed-forward path — two thirds of a block's parameters —
                    # unregularised. nn.Dropout holds no parameters and is the identity
                    # at p=0.0, so the default configuration is unchanged.
                    "d1": nn.Dropout(dropout),
                    "d2": nn.Dropout(dropout),
                    "d3": nn.Dropout(dropout),
                }
            )
            for _ in range(n_blocks)
        )
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, frames, audio_dim) -> (B, canvas_len, vocab_size) logits."""
        a = self.audio_proj(audio)
        x = self.queries.expand(audio.size(0), -1, -1)
        for b in self.blocks:
            h = b["n1"](x)
            x = x + b["d1"](b["sa"](h, h, h, need_weights=False)[0])
            h = b["n2"](x)
            x = x + b["d2"](b["xa"](h, a, a, need_weights=False)[0])
            x = x + b["d3"](b["ff"](b["n3"](x)))
        return self.head(self.norm_out(x))


def build_audio_decoder(
    cfg: DictConfig | dict, vocab_size: int, canvas_len: int
) -> AudioDecoder:
    """Build mu_psi from the `model.audio_decoder` node of the config.

    `audio_dim` defaults to the large-v3 encoder width — the width of the cached latents,
    so it follows the checkpoint the features were precomputed with, not the model being
    trained. Pass the DataModule's `d_model` explicitly when they can differ.

    vocab_size and canvas_len are arguments rather than config keys because they are
    properties of the run that the dataset decides: the compact vocabulary is measured
    from the corpus, and the canvas width is shared with `collate`.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"model.audio_decoder is missing {missing}; it needs {list(REQUIRED_KEYS)} "
            f"(plus optional d_ff, dropout and audio_dim, default {WHISPER_DIM})"
        )

    defaults = OmegaConf.create({"audio_dim": WHISPER_DIM, "d_ff": None, "dropout": 0.0})
    cfg = OmegaConf.merge(defaults, cfg)
    return AudioDecoder(
        audio_dim=cfg.audio_dim,
        vocab_size=vocab_size,
        canvas_len=canvas_len,
        hidden=cfg.hidden,
        n_heads=cfg.n_heads,
        n_blocks=cfg.n_blocks,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    )
