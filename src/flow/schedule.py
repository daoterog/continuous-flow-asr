"""The time reparameterization: tau, its inverse, and the curves behind it.

Not metrics, though one of them is measured with one. ``BinnedTokenError`` reports P_e(t);
this module turns a P_e curve into tau (Eq. 25), inverts it, and hands training and the
sampler the same u -> t map. Two curves can feed it:

  * what the interpolant alone does to a target, available before a single step of
    training. ``src.flow.recovery`` integrates this exactly and ``exact_tau_inverse``
    below is the whole construction; ``sweep_corruption`` is the Monte Carlo form it
    replaced, kept because the scripts that validate the exact curve need something
    independent to contest it with.
  * the model's own P_e, via MeasuredTau — which moves left as the denoiser learns to read
    the audio and recovers tokens the corrupted field alone could not.

One `tau_*` strategy drives each: `SimulatedTau` for the first, `MeasuredTau` for the
second. They decide when tau is re-derived and hand the model the new one.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.flow.recovery import recovery_curve

# Values of t the exact curve is evaluated at before inverting. Not a precision tradeoff
# the way `schedule.num_bins` is for the measured curve: the curve is exact at every point,
# so this only sets how finely tau^-1 is interpolated between knots. Against a 40,001-point
# build it is within 1e-6 in t, for 12-36 ms.
EXACT_GRID_POINTS = 4001


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

    Inverting it means reading the graph with the axes swapped, interpolating with tau as
    x and t as y. Two details that needs:
      * tau is pinned at 0 across the dead zone, and interpolation needs x strictly
        increasing. Keeping the LAST t of each flat run makes tau^-1(0) the right edge of
        that zone instead of spreading samples across a region where nothing is decided.
      * below flat_eps tau is indistinguishable from 0 — part sampling noise, part real
        but negligible recovery — and the monotone fix above has already frozen those in
        as genuine-looking steps. Clamping to exactly 0 is what stops them becoming knots.

    Neither end needs anchoring when the curve is swept at t the schedule did not choose:
    tau is normalised by the lowest bin's error so tau(lowest) == 0 by construction, and
    the highest bin sits at ~1. Measured on a trained checkpoint with a 150-bin sweep,
    tau^-1(0) lands at 0.030 and tau^-1(1) at 0.997 on their own, and tau clears flat_eps
    within 5 bins. A wide flat run then means the model really is making no progress there,
    and skipping it is the point.

    See scripts/schedule/explain_tau.py, which rebuilds this with np.interp and plots each stage.

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
    tau_inverse = tau_inverse_from_knots(tau_k, t_k)
    tau_inverse.tau = tau
    tau_inverse.ts = ts
    return tau_inverse


def tau_inverse_from_knots(tau_k: np.ndarray, t_k: np.ndarray) -> Callable:
    """t = tau^-1(u) from its knots alone: piecewise-linear, `tau_k` strictly increasing.

    What `make_tau_inverse` builds once it has the knots, split out so a curve saved with
    a checkpoint (`save_tau`) can be rebuilt exactly, including a measured one that no
    config could reproduce. The returned callable carries `knots`.
    """
    tau_k = np.asarray(tau_k, dtype=np.float64)
    t_k = np.asarray(t_k, dtype=np.float64)
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

    tau_inverse.knots = (tau_k, t_k)
    return tau_inverse


def exact_tau_inverse(
    vocab_size: int,
    alpha_fn: Callable,
    beta_fn: Callable,
    gamma_fn: Callable | None = None,
    mu_accuracy: float = 0.0,
    points: int = EXACT_GRID_POINTS,
) -> Callable:
    """Build t = tau^-1(u) for the interpolant path, with no model and no sampling in it.

    The prior tau, and the only thing `SimulatedTau` ever rebuilds. It needs no tokens and
    no datamodule: recovery under the interpolant depends on the vocabulary size and the
    coefficients alone, which is why `sweep_corruption` was drawing real targets only to
    average the vocabulary out of them again.

    Two arguments `make_tau_inverse` defaults differently for the measured curve:

      * `p_e0` is the path's own error at t=0, not the chance rate `1 - 1/V`. They agree
        when gamma(0) == 0, but `audio_cond` has gamma(0) == 1, so mu is already in the
        field at t=0 and recovery starts above chance — under the chance normaliser tau(0)
        comes out at 0.012 rather than 0 at mu accuracy 0.9, and drifts further up as mu
        learns. This is what `MeasuredTau` already passes.
      * `flat_eps` is 0. It exists to stop sampling noise near tau=0 being frozen in as
        knots, and the exact curve has none: it is strictly increasing from t=0 up to the
        point where recovery saturates, so tau^-1(0) can be read at face value.

    The saturation at the top is real and is left alone — recovery reaches 1.0 to machine
    precision around t=0.93 at V=1193, so tau is flat above it and many u map to the same
    t. `generation_grid` collapses those, which is still the right handling.
    """
    ts = np.linspace(0.0, 1.0, points)
    recovery = recovery_curve(ts, vocab_size, alpha_fn, beta_fn, gamma_fn, mu_accuracy)
    return make_tau_inverse(
        ts, recovery, vocab_size, flat_eps=0.0, p_e0=float(1.0 - recovery[0])
    )


def generation_grid(
    tau_inverse: Callable, num_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """(t_n, u_n) with t_n = tau^-1(u_n), u_n = n/N — the sampler's schedule, uniform in tau.

    Both are returned because a denoiser trained with `flow.time_input: uniform` is fed
    u_n while the state is still stepped in t_n, and u_n is what this starts from — no need
    to map t back.

    Training and generation have to share this. Once the model is trained under the
    reparameterization it has seen almost nothing below the dead zone, so integrating on
    a uniform-in-t grid afterwards would run it far out of distribution.

    Repeated t are dropped, so `num_steps` is a budget rather than a promise. Wherever tau
    is flat, several u map to the same t, and the integrator would re-run the denoiser on a
    state it has already seen — measured on a trained checkpoint, five of sixteen steps all
    landed on t=1.000 and integrated nothing. Collapsing them skips the dead interval and
    spends the steps that remain on t that actually differ.

    This is `schedule_bin_edges(tau_inverse, num_steps)` up to that collapse: the two share
    a construction, but a bin may legitimately be empty while a step may not be a no-op.
    """
    u = np.linspace(0.0, 1.0, num_steps + 1)
    grid = np.asarray(tau_inverse(u), dtype=np.float64)
    # Of each run of repeated t, the first u is the one kept.
    keep = np.concatenate([[True], np.diff(grid) > 0])
    grid, u = grid[keep], u[keep]
    if grid.size < 2:  # a constant tau leaves nothing to integrate along
        return np.array([float(grid[0]) if grid.size else 0.0, 1.0]), np.array([0.0, 1.0])
    return grid, u
