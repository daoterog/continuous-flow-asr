"""Training-time augmentation of the audio latents."""

import torch
from omegaconf import DictConfig


def mask_latents(audio: torch.Tensor, spec: DictConfig | dict | None) -> torch.Tensor:
    """SpecAugment on Whisper latents: zero random spans of frames and of channels.

    `train.audio_mask`, e.g. {time_masks: 2, time_width: 0.1, feature_masks: 2,
    feature_width: 0.1}: that many masks per utterance along each axis, each up to that
    fraction of the axis wide, drawn independently per row. None or an empty spec returns
    `audio` untouched.

    Every round-1 arm fits train.100 almost exactly at the transition (train loss ~0.001
    against ~0.1 on val): the audio is enough to recognise a training utterance and recite
    its transcript. Hiding part of it on every step is the standard ASR countermeasure,
    and it acts on that path specifically rather than on the whole network as dropout does.

    Widths are fractions of the padded length, so a short utterance in a long batch loses
    proportionally more of its real frames; the masks are drawn on the GPU, per batch.
    """
    if not spec:
        return audio
    b, frames, dim = audio.shape
    keep = torch.ones(b, frames, dim, dtype=torch.bool, device=audio.device)
    for axis, size, count, width in (
        (1, frames, int(spec.get("time_masks", 0)), float(spec.get("time_width", 0.0))),
        (2, dim, int(spec.get("feature_masks", 0)), float(spec.get("feature_width", 0.0))),
    ):
        pos = torch.arange(size, device=audio.device)[None, :]
        for _ in range(count):
            w = (torch.rand(b, 1, device=audio.device) * width * size).long()
            start = (torch.rand(b, 1, device=audio.device) * (size - w + 1)).long()
            span = (pos >= start) & (pos < start + w)  # (b, size)
            keep &= ~(span[:, :, None] if axis == 1 else span[:, None, :])
    return audio * keep


def audio_drop(t: torch.Tensor, spec: DictConfig | dict | None) -> torch.Tensor | None:
    """(B,) which rows train with the audio taken away, or None when nothing is dropped.

    `train.audio_dropout`, e.g. {p: 0.2, schedule: constant} or {p: 0.4, schedule: linear}:
    each row is dropped with probability p, or p * t under `linear`, so the nearer the
    transcript the field is, the more often it has to be solved from the field alone.

    In training the field at high t is built from the true transcript, and the audio is
    the true audio: the two always agree, so nothing rewards reading the text, and every
    round-1 checkpoint probed leans on the audio even where the field is nearly clean. A
    dropped row has only the field, which is language modelling — the coherence the late
    sampling steps are meant to supply.
    """
    if not spec or float(spec.get("p", 0.0)) <= 0.0:
        return None
    p = float(spec["p"])
    schedule = str(spec.get("schedule", "constant"))
    if schedule == "constant":
        prob = torch.full_like(t, p)
    elif schedule == "linear":
        prob = p * t
    else:
        raise ValueError(
            f"train.audio_dropout.schedule must be constant or linear, got {schedule!r}"
        )
    return torch.rand_like(t) < prob
