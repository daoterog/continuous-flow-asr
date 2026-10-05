"""The two-time denoiser: D(x_s, s, t), the endpoint as if jumping straight from s to t.

Distillation needs a model that answers "given the state at s, what is the endpoint if I
jump to t", so it takes two absolute times rather than one. The ordinary denoiser is the
case s == t, which is how FMLM trains it: part of every batch sits on that diagonal
against the teacher's one-step output, the rest off it against a two-step rollout.

DRAX's Transformer is vendored unmodified (CC-BY-NC, Meta / aiOla), so the second time is
added around it rather than inside it. Its `forward` only ever passes `time` to
`self.time_embedding`, plus a `torch.cat([time, time], dim=0)` in the CFG branch that is
indifferent to the trailing dimension — so swapping the embedder for one that reads a
(B, 2) tensor is enough, and the vendored file stays byte-identical.
"""

from src.constants import WHISPER_DIM

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .denoiser import REQUIRED_KEYS
from .drax.transformer import TimestepEmbedder, Transformer


class TwoTimeEmbedder(nn.Module):
    """Embeds (s, t) into the one conditioning vector adaLN expects.

    Two independent `TimestepEmbedder`s summed, with the end branch's output projection
    zero-initialised — the same trick DRAX uses on `adaLN_modulation`. At init the sum is
    exactly the start branch, so a fresh two-time model *is* the one-time model and a
    teacher's weights transfer into it unchanged. The dependence on t is learned from zero
    rather than from noise.

    Absolute (s, t) rather than (s, t - s): it is what FMLM conditions on, and it keeps
    s == t meaning "no jump" without the model having to represent a gap of zero specially.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """Build both branches and silence the end one."""
        super().__init__()
        self.start = TimestepEmbedder(hidden_size, frequency_embedding_size)
        self.end = TimestepEmbedder(hidden_size, frequency_embedding_size)
        final = self.end.mlp[-1]
        final.weight.data.zero_()
        final.bias.data.zero_()

    def forward(self, time: Tensor) -> Tensor:
        """(B, 2) of (start, end) -> (B, hidden). A plain (B,) is read as start only."""
        if time.ndim == 1:
            return self.start(time)
        return self.start(time[:, 0]) + self.end(time[:, 1])


class TwoTimeTransformer(Transformer):
    """DRAX's Transformer conditioned on a start and an end time.

    `forward` keeps the base signature and adds `time_end`. Left off, it repeats `time`, so
    the model behaves as the ordinary denoiser and every existing call site — the sampler,
    the metrics, `flow_step` — works unchanged.
    """

    def __init__(self, vocab_size: int, masked: bool, config: DictConfig) -> None:
        """Build the base model, then replace its time embedding with the two-time one."""
        super().__init__(vocab_size=vocab_size, masked=masked, config=config)
        self.time_embedding = TwoTimeEmbedder(hidden_size=config.cond_dim)

    def forward(
        self, x_t: Tensor, time: Tensor, time_end: Tensor | None = None, **kwargs: object
    ) -> Tensor:
        """Predict the endpoint at `time_end` from the state at `time`."""
        if time_end is None:
            time_end = time
        packed = torch.stack([time, time_end], dim=-1)
        return super().forward(x_t=x_t, time=packed, **kwargs)


def build_two_time_denoiser(
    cfg: DictConfig | dict, vocab_size: int
) -> TwoTimeTransformer:
    """Build the two-time denoiser from the same `model.denoiser` node as the one-time one.

    Takes the identical config so a distillation run can reuse its teacher's geometry
    without a second block of YAML; the only difference is the extra time embedder, which
    is sized off `cond_dim` like the first.
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
    return TwoTimeTransformer(vocab_size=vocab_size, masked=False, config=cfg)


def load_teacher_weights(student: TwoTimeTransformer, teacher_state: dict) -> None:
    """Initialise a two-time student from a one-time denoiser's state_dict.

    Only the time embedding moved: `time_embedding.mlp.*` became
    `time_embedding.start.mlp.*`. The end branch keeps its zero init and is the only thing
    not supplied by the teacher, so `strict=False` here hides nothing else — the call
    checks that is all that is missing.
    """
    remapped = {
        k.replace("time_embedding.mlp.", "time_embedding.start.mlp."): v
        for k, v in teacher_state.items()
    }
    missing, unexpected = student.load_state_dict(remapped, strict=False)
    unexpected = [k for k in unexpected]
    stray = [k for k in missing if not k.startswith("time_embedding.end.")]
    if stray or unexpected:
        raise RuntimeError(
            f"teacher weights do not fit the two-time student: missing {stray}, "
            f"unexpected {unexpected}"
        )
