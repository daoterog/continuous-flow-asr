"""The time reparameterization: tau, its inverse, and the corruption sweep behind it.

Not metrics, though it is measured with one. ``BinnedTokenError`` reports P_e(t); this
module turns a P_e curve into tau (Eq. 25), inverts it, and hands training and the
sampler the same u -> t map. Two curves can feed it:

  * the model-free corruption sweep below — what the interpolant alone does to a target,
    available before a single step of training;
  * the model's own P_e, via MeasuredTau — which moves left as the denoiser
    learns to read the audio and recovers tokens the corrupted field alone could not.

One `tau_*` strategy drives each: `SimulatedTau` for the sweep, `MeasuredTau` for the
model's own curve. They decide when tau is re-derived and hand the model the new one.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


def simulate_mu(
    tokens: torch.Tensor, vocab_size: int, accuracy: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Swaps tokens for random ones with probability 1-accuracy, then one-hot encodes."""
    right = torch.rand(tokens.shape, device=tokens.device, generator=generator) < accuracy
    wrong = torch.randint(
        0, vocab_size, tokens.shape, device=tokens.device, generator=generator
    )
    return F.one_hot(torch.where(right, tokens, wrong), vocab_size).float()


@torch.no_grad()
def corrupt_and_decode(
    tokens: torch.Tensor,
    vocab_size: int,
    t: float,
    alpha_fn: Callable | None = None,
    beta_fn: Callable | None = None,
    gamma_fn: Callable | None = None,
    mu_accuracy: float = 0.0,
    chunk: int = 8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The interpolant with no model in the loop: argmax of alpha*x0 + beta*x1 [+ gamma*mu].

    Per canvas position:
        - the **true** slot lands at beta(t) + alpha(t)*N(0,1)
        - and for the others V-1 slots at alpha(t)*N(0,1)

    argmax recovers the token only when the spike beats the largest of those V-1 draws.

    When using mu we need to consider it's accuracy, because the interpolant depends on
    mu's output.

    Chunked because a (B, L, V) float field is large and randn_like allocates a second one.
    """
    alpha_fn = alpha_fn or (lambda t: 1 - t)
    beta_fn = beta_fn or (lambda t: t)
    out = []
    for i in range(0, tokens.size(0), chunk):
        piece = tokens[i : i + chunk]
        x1 = F.one_hot(piece, vocab_size).float()
        field = alpha_fn(t) * torch.randn(
            x1.shape, device=x1.device, generator=generator
        ) + beta_fn(t) * x1
        if gamma_fn is not None:
            field = field + gamma_fn(t) * simulate_mu(
                piece, vocab_size, mu_accuracy, generator
            )
        out.append(field.argmax(dim=-1))
    return torch.cat(out)


@torch.no_grad()
def sweep_corruption(
    tokens: torch.Tensor,
    vocab_size: int,
    ts: list[float],
    alpha_fn: Callable | None = None,
    beta_fn: Callable | None = None,
    gamma_fn: Callable | None = None,
    mu_accuracy: float = 0.0,
    chunk: int = 8,
) -> list[float]:
    """Token recovery under the interpolant alone, one value per t in `ts`.

    This is the prior tau is built from before any training: how much of the transcript
    the field carries on its own. Pass the schedules the run will actually train with, or
    tau describes a different probability path than the model sees.

    The mean runs over every canvas position, padding included, and still matches the
    model-measured P_e that BinnedTokenError reports over real tokens only: recovery here
    depends on t and V alone — the spike sits on whichever token is true, so a pad position
    is recovered with exactly the same probability as a word. Averaging over all positions
    and over real ones agree in expectation, so the prior and the refresh stay comparable.
    """
    return [
        (
            corrupt_and_decode(
                tokens, vocab_size, t, alpha_fn, beta_fn, gamma_fn, mu_accuracy, chunk
            )
            == tokens
        )
        .float()
        .mean()
        .item()
        for t in ts
    ]


def tau_from_recovery(
    recovery: np.ndarray | list[float], vocab_size: int, p_e0: float | None = None
) -> np.ndarray:
    """tau(t) = 1 - P_e(t)/P_e(0), cumulative decoding progress in [0, 1]."""
    recovery = np.asarray(recovery, dtype=np.float64)
    p_e = 1.0 - recovery
    if p_e0 is None:
        p_e0 = 1.0 - 1.0 / vocab_size
    tau = np.clip(1.0 - p_e / max(p_e0, 1e-12), 0.0, 1.0)
    return np.maximum.accumulate(tau)


def make_tau_inverse(
    ts: np.ndarray | list[float],
    recovery: np.ndarray | list[float],
    vocab_size: int,
    flat_eps: float = 1e-3,
    p_e0: float | None = None,
) -> Callable:
    """Build t = tau^-1(u), the map from a uniform draw to a flow time.

    This is the per-token error rate.

    Two details the naive inversion gets wrong:
      * tau is ~0 across the whole dead zone (t < ~0.65 at Whisper's vocabulary). Keeping
        the LAST t of each flat run makes tau^-1(0) the right edge of that zone instead
        of spreading samples linearly across a region where nothing is decided.
      * a finite sample makes tau wobble by ~1e-4 down there, and those wobbles survive
        the monotone fix as genuine-looking steps; flat_eps clamps them to exactly 0.

    The returned callable carries `tau`, `ts` and `knots` as attributes, for plotting and
    for ``schedule_bin_edges``.

    A tensor argument is interpolated on its own device. Training calls this once per
    step, so routing it through numpy would mean a device-to-host copy — and therefore a
    full CUDA sync, blocking on every kernel still in flight — inside the hot loop.
    """
    ts = np.asarray(ts, dtype=np.float64)
    tau = tau_from_recovery(recovery, vocab_size, p_e0=p_e0)
    tau[tau < flat_eps] = 0.0

    keep = np.concatenate([np.diff(tau) > 0, [True]])  # last index of every flat run
    tau_k, t_k = tau[keep], ts[keep]
    knots: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}  # uploaded once per device

    def _knots_on(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in knots:
            # float64 so the interpolation matches np.interp to ~1e-9 rather than the
            # ~4e-5 float32 accumulates; it is a handful of values, so the cost is nil.
            knots[device] = (
                torch.as_tensor(tau_k, dtype=torch.float64, device=device),
                torch.as_tensor(t_k, dtype=torch.float64, device=device),
            )
        return knots[device]

    def tau_inverse(u: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Map u: (B,) uniform in [0, 1] to t: (B,) flow times."""
        if not torch.is_tensor(u):
            return np.interp(np.asarray(u), tau_k, t_k)

        xk, yk = _knots_on(u.device)
        if xk.numel() == 1:  # tau never rises: every draw lands on the one knot
            return torch.full_like(u, float(t_k[0]))

        # Piecewise-linear, matching np.interp including its clamp outside the knots:
        # below the first knot w goes negative and above the last it exceeds 1, and the
        # clamp pins both to the endpoint values.
        x = u.detach().double()
        idx = torch.searchsorted(xk, x.contiguous(), right=True).clamp(1, xk.numel() - 1)
        lo, hi = xk[idx - 1], xk[idx]
        w = ((x - lo) / (hi - lo).clamp_min(torch.finfo(torch.float64).tiny)).clamp(0, 1)
        return torch.lerp(yk[idx - 1], yk[idx], w).to(u.dtype)

    tau_inverse.tau = tau
    tau_inverse.ts = ts
    tau_inverse.knots = (tau_k, t_k)
    return tau_inverse


def generation_grid(tau_inverse: Callable, num_steps: int) -> np.ndarray:
    """t_n = tau^-1(n/N) — the sampler's schedule, uniform in tau rather than in t.

    Training and generation have to share this. Once the model is trained under the
    reparameterization it has seen almost nothing below the dead zone, so integrating on
    a uniform-in-t grid afterwards would run it far out of distribution.

    Note this is schedule_bin_edges(tau_inverse, num_steps): the sampler's grid and the
    metrics' bin edges are the same construction, which is why one bin is one step.
    """
    return tau_inverse(np.linspace(0.0, 1.0, num_steps + 1))
