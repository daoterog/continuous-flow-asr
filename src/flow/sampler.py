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

Two integrators, chosen by `sampling.method`:

`sample_tokens` ("interpolant") rebuilds the interpolant at the next time with x0 held
fixed. For affine g that is algebraically identical to one Euler step of the drift, and for
the quadratic g of three_term it is better than Euler: it uses the exact g(t+h) rather than
the first-order g(t) + h g'(t). It has no 1/(1-t) to blow up at the end, and it reads the
transcript off the denoiser's last prediction.

`sample_tokens_euler` ("euler") is Algorithm 6 of the paper, verbatim: step the state along
the drift, x_{n+1} = x_n + (t_{n+1} - t_n) b_n, and decode the final *state* rather than the
last prediction. It is the reference formulation, so it is the one to reproduce the paper's
numbers with; the two agree exactly on linear and audio_cond and differ at O(h^2) on
three_term.
"""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F


def drift(
    x_t: torch.Tensor,
    denoised: torch.Tensor,
    t: float,
    mu: torch.Tensor | None = None,
    waypoint: float = 0.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """b_t(x | a), as the paper writes it per variant.

        linear      (D - x) / (1 - t)
        audio_cond  (D - x) / (1 - t)                    the mu term cancels
        three_term  (D - x) / (1 - t) + 4c(1 - t) mu

    `waypoint` is that 4c — `waypoint_strength` of the gamma spec — and is 0 for the two
    variants without the term, so the whole drift is these three lines with no gamma and
    no derivative of it anywhere.
    """
    velocity = (denoised - x_t) / max(1.0 - t, eps)
    if waypoint and mu is not None:
        velocity = velocity + waypoint * (1.0 - t) * mu
    return velocity


def denoiser_times(
    times: list[float], time_grid: np.ndarray | list[float] | None
) -> list[float]:
    """The denoiser's time input at each grid point: `time_grid` when given, else t.

    Under `flow.time_input: uniform` the denoiser was trained on the uniform draw u, not on
    t = tau^-1(u), so the sampler hands it the u each grid point was built from.
    """
    if time_grid is None:
        return times
    inputs = [float(v) for v in time_grid]
    if len(inputs) != len(times):
        raise ValueError(
            f"time_grid has {len(inputs)} points but the grid has {len(times)}"
        )
    return inputs


def _endpoint(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    start: float,
    end: float,
    audio: torch.Tensor,
    vocab_size: int,
    hard: bool,
) -> torch.Tensor:
    """One two-time call: the endpoint the model predicts for the jump `start -> end`.

    `D(x_s, s, t)` is an estimate of x1, not of the state at t — which is why the last step
    of both two-time samplers can decode it directly.
    """
    batch = audio.size(0)
    times = {
        name: torch.full((batch,), value, device=audio.device, dtype=torch.float32)
        for name, value in (("startpoint", start), ("endpoint", end))
    }
    logits = model(x_t=x_t, audio_embeddings=audio, **times)
    if hard:
        return F.one_hot(logits.argmax(dim=-1), vocab_size).to(logits.dtype)
    return F.softmax(logits, dim=-1)


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
    waypoint: float = 0.0,
    time_grid: np.ndarray | list[float] | None = None,
) -> torch.Tensor:
    """Integrate from noise to (B, canvas_len) token ids.

    `time_grid` is what the denoiser's time input reads at each point of `grid`; None
    feeds it `grid` itself. See `denoiser_times`.

    `waypoint` is unused — this integrator never forms the drift, it rebuilds the
    interpolant from gamma_fn directly — and is accepted only so the two samplers share one
    signature behind `pl_module.SAMPLERS`.
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
    inputs = denoiser_times(times, time_grid)
    x_t = state(times[0], x1_init)
    logits = None
    for step, (t, t_next) in enumerate(pairwise(times)):
        time = torch.full((batch,), inputs[step], device=audio.device, dtype=torch.float32)
        logits = denoiser(x_t=x_t, time=time, audio_embeddings=audio)
        x1_hat = (
            F.one_hot(logits.argmax(dim=-1), vocab_size).to(logits.dtype)
            if hard
            else F.softmax(logits, dim=-1)
        )
        if step < len(times) - 2:
            # Avoid allocating unnecessary space in the GPU
            x_t = state(t_next, x1_hat)

    return logits.argmax(dim=-1)



@torch.no_grad()
def sample_tokens_euler(
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
    waypoint: float = 0.0,
    eps: float = 1e-5,
    time_grid: np.ndarray | list[float] | None = None,
) -> torch.Tensor:
    """Algorithm 6: Euler-step the drift, then decode the final state.

    `time_grid` as in `sample_tokens`.

    Differs from `sample_tokens` in both halves of the loop. The state is advanced by
    `x + (t_next - t) b` instead of being rebuilt from x0 at the new time, and the
    transcript comes from `argmax(x_{t_N})` — the integrated state — rather than from the
    denoiser's last prediction. The two coincide only if the grid ends at t=1, where the
    state is the endpoint term alone.
    """
    batch = audio.size(0)
    x0 = torch.randn(
        batch, canvas_len, vocab_size, device=audio.device, generator=generator
    )
    times = [float(t) for t in grid]

    x_t = alpha_fn(times[0]) * x0
    if x1_init is not None:
        x_t = x_t + beta_fn(times[0]) * x1_init
    if gamma_fn is not None and mu is not None:
        x_t = x_t + gamma_fn(times[0]) * mu

    inputs = denoiser_times(times, time_grid)
    for step, (t, t_next) in enumerate(pairwise(times)):
        time = torch.full((batch,), inputs[step], device=audio.device, dtype=torch.float32)
        logits = denoiser(x_t=x_t, time=time, audio_embeddings=audio)
        x1_hat = (
            F.one_hot(logits.argmax(dim=-1), vocab_size).to(logits.dtype)
            if hard
            else F.softmax(logits, dim=-1)
        )
        x_t = x_t + (t_next - t) * drift(x_t, x1_hat, t, mu, waypoint, eps)

    return x_t.argmax(dim=-1)


@torch.no_grad()
def sample_map_interpolant(
    model: torch.nn.Module,
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
    """Integrate a two-time model, rebuilding the interpolant at each step.

    The two-time analogue of `sample_tokens`: the endpoint estimate comes from
    `D(x, s, t_next)` instead of `D(x, s)`, and the state is rebuilt from the *same* x0 at
    the next time rather than advanced. x0 held fixed is what distinguishes this from
    `sample_map`; the error of one step does not carry into the next.

    No `time_grid` and no `waypoint`. Distillation conditions both model inputs on real flow
    time — `FlowMapping` refuses a teacher trained any other way — so there is no u to feed,
    and the waypoint term belongs to a gamma this path already rules out.
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
    x1_hat = None
    for step, (t, t_next) in enumerate(pairwise(times)):
        x1_hat = _endpoint(model, x_t, t, t_next, audio, vocab_size, hard)
        if step < len(times) - 2:
            x_t = state(t_next, x1_hat)
    return x1_hat.argmax(dim=-1)


@torch.no_grad()
def sample_map(
    model: torch.nn.Module,
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
    eps: float = 1e-5,
) -> torch.Tensor:
    """Integrate a two-time model by advancing the state — FMLM's `generate_samples`.

    The step is the Euler map the distillation rollout already trains against:

        x <- state * x + pred * D(x, s, t_next),
        state = (1 - t_next) / (1 - s),   pred = (t_next - s) / (1 - s)

    which is `x + h (D - x) / (1 - s)` rearranged, so it is the drift step of
    `sample_tokens_euler` written in the form `map_step` uses. The difference from
    `sample_map_interpolant` is only that x carries forward instead of being rebuilt.

    The last step returns the prediction rather than the integrated state (algo.py:1548).
    That matches `sample_tokens` and differs from `sample_tokens_euler`, which decodes the
    state — a difference worth keeping in mind, since it is the only thing separating those
    two one-time samplers.
    """
    batch = audio.size(0)
    x_t = alpha_fn(grid[0]) * torch.randn(
        batch, canvas_len, vocab_size, device=audio.device, generator=generator
    )
    if x1_init is not None:
        x_t = x_t + beta_fn(grid[0]) * x1_init
    if gamma_fn is not None and mu is not None:
        x_t = x_t + gamma_fn(grid[0]) * mu

    times = [float(t) for t in grid]
    x1_hat = None
    for step, (t, t_next) in enumerate(pairwise(times)):
        x1_hat = _endpoint(model, x_t, t, t_next, audio, vocab_size, hard)
        if step < len(times) - 2:
            remaining = max(1.0 - t, eps)
            x_t = ((1.0 - t_next) / remaining) * x_t + ((t_next - t) / remaining) * x1_hat
    return x1_hat.argmax(dim=-1)
