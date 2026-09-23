"""Integrate the flow from noise to a transcript — what inference actually does.

The binned metrics score one denoising step from an x_t built out of the *target*, so at
high t they largely measure the input: the corruption sweep recovers ~75% of tokens at
t=0.8 with no model at all. Sampling starts from x0 ~ N(0, I) with no target anywhere and
integrates the grid, so its WER is the number comparable to the leaderboard.

The model is a denoiser D (it predicts x1), and the drift follows from it. For the affine
path x_t = a(t) x0 + b(t) x1 + g(t) mu, eliminating x0 gives

    b_t(x) = (a'/a) x + (b' - a'b/a) D_t(x) + (g' - a'g/a) mu

and with a = 1-t, b = t this is

    b_t(x) = (D_t(x) - x) / (1 - t)  +  [g'(t) + g(t)/(1-t)] mu

so the three variants read:

    linear      g = 0            ->  (D - x)/(1 - t)
    audio_cond  g = 1 - t        ->  (D - x)/(1 - t)          the mu term cancels
    three_term  g = 4c t (1-t)   ->  (D - x)/(1 - t) + 4c(1-t) mu

`sample_tokens` integrates by rebuilding the interpolant at the next time with x0 held
fixed, which for affine g is algebraically identical to one Euler step of that drift, and
for the quadratic g of three_term is better than Euler: it uses the exact g(t+h) rather
than the first-order g(t) + h g'(t). It also has no 1/(1-t) to blow up at the end.
`drift` is provided so the equivalence can be checked directly.
"""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from normalizer import data_utils

from src.metrics import corpus_wer, ids_to_text


def drift(
    x_t: torch.Tensor,
    denoised: torch.Tensor,
    t: float,
    gamma_fn: Callable | None = None,
    mu: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """b_t(x) for the linear path, including the waypoint term when gamma is present.

    Kept for cross-checking the sampler and for anyone who prefers the ODE view; the
    sampler itself does not call it. g'(t) is taken numerically so any gamma_fn works.
    """
    one_minus_t = max(1.0 - t, eps)
    velocity = (denoised - x_t) / one_minus_t
    if gamma_fn is not None and mu is not None:
        g = gamma_fn(t)
        g_prime = (gamma_fn(min(t + eps, 1.0)) - gamma_fn(max(t - eps, 0.0))) / (2 * eps)
        velocity = velocity + (g_prime + g / one_minus_t) * mu
    return velocity


@torch.no_grad()
def sample_tokens(
    denoiser: torch.nn.Module,
    audio: torch.Tensor,
    grid: np.ndarray | list[float],
    vocab_size: int,
    canvas_len: int,
    alpha_fn: Callable,
    beta_fn: Callable,
    gamma_fn: Callable | None = None,
    mu: torch.Tensor | None = None,
    hard: bool = False,
    x1_init: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integrate from noise to (B, canvas_len) token ids.

    grid: increasing flow times, normally `generation_grid(tau_inverse, num_steps)` — the
    same construction as the metric bin edges, which is why one bin is one step.

    x0 is drawn once and held fixed for the whole trajectory; that is what makes the
    rebuild equal to an Euler step of the drift.

    x1_init seeds the endpoint term for the very first state. It matters because the tau
    grid does not start at 0 — it starts at the right edge of the dead zone, typically
    ~0.2 — so training's state there already contains beta(t0) * x1, while the sampler
    has no estimate yet. Left as None that term is simply dropped, which is exact only
    when beta(t0) = 0. Passing mu supplies the audio-only estimate of x1, which is the
    best guess available before the denoiser has been called at all.

    hard=True feeds back a one-hot argmax instead of the softmax. The softmax is closer
    to what training showed the model (x1 is a point on the simplex), so it is default.
    """
    batch = audio.size(0)
    x0 = torch.randn(
        batch, canvas_len, vocab_size, device=audio.device, generator=generator
    )

    def state(t: float, x1: torch.Tensor | None) -> torch.Tensor:
        out = alpha_fn(t) * x0
        if x1 is not None:
            out = out + beta_fn(t) * x1
        if gamma_fn is not None and mu is not None:
            out = out + gamma_fn(t) * mu
        return out

    times = [float(t) for t in grid]
    x_t = state(times[0], x1_init)
    logits = None
    for t, t_next in pairwise(times):
        time = torch.full((batch,), t, device=audio.device, dtype=torch.float32)
        logits = denoiser(x_t=x_t, time=time, audio_embeddings=audio)
        x1_hat = (
            F.one_hot(logits.argmax(dim=-1), vocab_size).to(logits.dtype)
            if hard
            else F.softmax(logits, dim=-1)
        )
        x_t = state(t_next, x1_hat)

    # At t=1 the state is exactly the last endpoint estimate, so its argmax is the sample.
    return logits.argmax(dim=-1)


@torch.no_grad()
def score_samples(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer: object,
    pad_id: int,
    vocab_map: object | None = None,
) -> tuple[float, float]:
    """(WER %, token error rate) for sampled ids against the references.

    Both sides go through the decode and normalisation the binned WER uses, so the number
    is comparable to what `scripts/score_eval.py` reports.
    """
    preds = ids_to_text(pred_ids, tokenizer, pad_id, vocab_map)
    refs = ids_to_text(target_ids, tokenizer, pad_id, vocab_map)
    pairs = [
        (r, p)
        for r, p in zip(refs, preds, strict=True)
        if data_utils.normalizer(r).strip()
    ]
    wer = corpus_wer([r for r, _ in pairs], [p for _, p in pairs]) if pairs else float("nan")
    token_error = (pred_ids != target_ids).float().mean().item()
    return wer, token_error
