"""Training-time augmentation of the audio latents."""

import torch
import torch.nn.functional as F
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


def corrupt_draft(
    x1: torch.Tensor,
    t: torch.Tensor,
    real: torch.Tensor,
    spec: DictConfig | dict | None,
    mu_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """The one-hot transcript the field is built from, some real tokens swapped for wrong ones.

    `train.draft_corruption`, e.g. {p: 0.1, source: mu}: each real token is replaced with
    probability p, or p * (1 - t) under `schedule: decreasing`, since a draft is worse the
    earlier it is. Only the field sees the swap; the loss still targets the true transcript.

    In training the field only ever holds the true transcript plus Gaussian noise, while at
    sampling time it holds the model's own draft, errors included. So the denoiser never
    learns to doubt the field, or to correct it from the audio and the words around it,
    which is what the late steps are for. The wrong tokens come from:

      uniform   any token. Easy to spot: a random token out of ~1,200 rarely fits the
                audio or its neighbours.
      mu        a sample from mu's softmax. mu reads the same audio, so its wrong guesses
                are the plausible confusions for this utterance — similar-sounding words,
                another subword split — the kind a draft makes, at no extra forward pass.
                A sample can land on the true token, so the rate of actual errors is below
                p, by mu's probability on the target (~0.35).
    """
    if not spec or float(spec.get("p", 0.0)) <= 0.0:
        return x1
    p = float(spec["p"])
    schedule = str(spec.get("schedule", "constant"))
    if schedule == "constant":
        prob = torch.full_like(t, p)
    elif schedule == "decreasing":
        prob = p * (1.0 - t)
    else:
        raise ValueError(
            f"train.draft_corruption.schedule must be constant or decreasing, got {schedule!r}"
        )
    hit = (torch.rand(real.shape, device=real.device) < prob[:, None]) & real

    source = str(spec.get("source", "uniform"))
    if source == "uniform":
        wrong = F.one_hot(
            torch.randint(0, x1.size(-1), real.shape, device=real.device), x1.size(-1)
        ).to(x1.dtype)
    elif source == "mu":
        if mu_logits is None:
            raise ValueError("train.draft_corruption.source=mu needs a run with mu")
        wrong = F.gumbel_softmax(mu_logits.detach().float(), hard=True).to(x1.dtype)
    else:
        raise ValueError(f"train.draft_corruption.source must be uniform or mu, got {source!r}")
    return torch.where(hit[..., None], wrong, x1)


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
