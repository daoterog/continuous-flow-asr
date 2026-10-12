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
