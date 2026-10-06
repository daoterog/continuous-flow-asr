"""The time reparameterization: tau, its inverse, and the curves behind it.

Not metrics, though one of them is measured with one. ``BinnedTokenError`` reports P_e(t);
this module turns a P_e curve into a `TimeReparametrization` (Eq. 25), which hands
training, the sampler and the binned metrics the same u <-> t map. Two curves can feed it:

  * what the interpolant alone does to a target, available before a single step of
    training. ``src.flow.recovery`` integrates this exactly and
    ``TimeReparametrization.exact`` below is the whole construction; ``sweep_corruption``
    is the Monte Carlo form it replaced, kept because the scripts that validate the exact
    curve need something independent to contest it with.
  * the model's own P_e, via MeasuredTau — which moves left as the denoiser learns to read
    the audio and recovers tokens the corrupted field alone could not.

One `tau_*` strategy drives each: `SimulatedTau` for the first, `MeasuredTau` for the
second. They decide when tau is re-derived and hand the model the new one.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.flow.recovery import recovery_curve

# What the map accepts and returns, matched: a float, an array, or a tensor on any device.
ArrayLike = float | np.ndarray | torch.Tensor

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


@dataclass(frozen=True, eq=False)
class TimeReparametrization:
    """u = tau(t), the fraction of decoding done by flow time t, and its inverse.

    Training draws u ~ U[0, 1] and builds the interpolant at t = tau^-1(u), so the batch
    lands where tokens are actually being decided rather than in the dead zone, where x_t
    carries nothing. The sampler walks the same map (`sampling_grid`), the binned metrics
    cut their bins along it (`bin_edges`), and a `flow.time_input: uniform` denoiser is fed
    u itself.

    The curve is piecewise-linear through its knots, and both directions read the same
    knots with the axes swapped. Build one with a named constructor:

        TimeReparametrization.identity()            no schedule: u == t
        TimeReparametrization.exact(V, a, b, g)     the interpolant's own curve, no model
        TimeReparametrization.from_recovery(...)    from a measured recovery curve
        TimeReparametrization.load(path)            as saved next to a checkpoint

    `tau_scheduler` strategies (`src.flow.tau_*`) decide when a new one replaces the old.
    """

    # Flow times, increasing. The first is tau^-1(0), the right edge of the dead zone; the
    # last is where recovery saturates.
    t_knots: np.ndarray
    # tau at those times, strictly increasing, in [0, 1].
    u_knots: np.ndarray
    # Where the curve came from: "identity", "simulated", "measured".
    source: str
    # The whole tau(t) curve before it was thinned to knots, for plotting. Only
    # `from_recovery` has one; the knots alone define the map.
    curve_t: np.ndarray | None = field(default=None, repr=False)
    curve_u: np.ndarray | None = field(default=None, repr=False)
    # Knots uploaded per device, so a tensor never round-trips through the host.
    _device_knots: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Hold the knots as float64 and check they describe a map."""
        t_k = np.asarray(self.t_knots, dtype=np.float64)
        u_k = np.asarray(self.u_knots, dtype=np.float64)
        object.__setattr__(self, "t_knots", t_k)
        object.__setattr__(self, "u_knots", u_k)
        if t_k.ndim != 1 or t_k.shape != u_k.shape or t_k.size == 0:
            raise ValueError(
                f"knots must be two non-empty 1-D arrays of one length, got "
                f"{t_k.shape} and {u_k.shape}"
            )
        if np.any(np.diff(u_k) <= 0) or np.any(np.diff(t_k) < 0):
            raise ValueError("u_knots must be strictly increasing and t_knots increasing")
        if t_k.min() < 0 or t_k.max() > 1 or u_k.min() < 0 or u_k.max() > 1:
            raise ValueError("knots must lie in [0, 1]")

    # -- constructors ---------------------------------------------------------------

    @classmethod
    def identity(cls) -> "TimeReparametrization":
        """No reparametrization: t is drawn uniformly, so u and t are the same number."""
        return cls(np.array([0.0, 1.0]), np.array([0.0, 1.0]), source="identity")

    @classmethod
    def from_recovery(
        cls,
        ts: np.ndarray | list[float],
        recovery: np.ndarray | list[float],
        vocab_size: int,
        *,
        flat_eps: float = 1e-3,
        p_e0: float | None = None,
        source: str = "measured",
    ) -> "TimeReparametrization":
        """tau from a token-recovery curve sampled at `ts` (`tau_from_recovery`, Eq. 25).

        Inverting it means reading the graph with the axes swapped, interpolating with tau
        as x and t as y. Two details that needs:
          * tau is pinned at 0 across the dead zone, and interpolation needs x strictly
            increasing. Keeping the LAST t of each flat run makes tau^-1(0) the right edge
            of that zone instead of spreading samples across a region where nothing is
            decided.
          * below flat_eps tau is indistinguishable from 0 — part sampling noise, part real
            but negligible recovery — and the monotone fix has already frozen those in as
            genuine-looking steps. Clamping to exactly 0 is what stops them becoming knots.

        Neither end needs anchoring when the curve is swept at t the schedule did not
        choose: tau is normalised by the lowest bin's error so tau(lowest) == 0 by
        construction, and the highest bin sits at ~1. Measured on a trained checkpoint with
        a 150-bin sweep, tau^-1(0) lands at 0.030 and tau^-1(1) at 0.997 on their own, and
        tau clears flat_eps within 5 bins. A wide flat run then means the model really is
        making no progress there, and skipping it is the point.

        See scripts/schedule/explain_tau.py, which rebuilds this with np.interp and plots
        each stage.
        """
        ts = np.asarray(ts, dtype=np.float64)
        tau = tau_from_recovery(recovery, vocab_size, p_e0=p_e0)
        tau[tau < flat_eps] = 0.0
        keep = np.concatenate([np.diff(tau) > 0, [True]])  # last index of every flat run
        return cls(ts[keep], tau[keep], source=source, curve_t=ts, curve_u=tau)

    @classmethod
    def exact(
        cls,
        vocab_size: int,
        alpha_fn: Callable,
        beta_fn: Callable,
        gamma_fn: Callable | None = None,
        mu_accuracy: float = 0.0,
        points: int = EXACT_GRID_POINTS,
    ) -> "TimeReparametrization":
        """tau for the interpolant path itself, with no model and no sampling in it.

        The prior tau, and the only thing `SimulatedTau` ever rebuilds. It needs no tokens
        and no datamodule: recovery under the interpolant depends on the vocabulary size and
        the coefficients alone, which is why `sweep_corruption` was drawing real targets
        only to average the vocabulary out of them again.

        Two arguments `from_recovery` defaults differently for the measured curve:

          * `p_e0` is the path's own error at t=0, not the chance rate `1 - 1/V`. They agree
            when gamma(0) == 0, but `audio_cond` has gamma(0) == 1, so mu is already in the
            field at t=0 and recovery starts above chance — under the chance normaliser
            tau(0) comes out at 0.012 rather than 0 at mu accuracy 0.9, and drifts further
            up as mu learns. This is what `MeasuredTau` already passes.
          * `flat_eps` is 0. It exists to stop sampling noise near tau=0 being frozen in as
            knots, and the exact curve has none: it is strictly increasing from t=0 up to
            the point where recovery saturates, so tau^-1(0) can be read at face value.

        The saturation at the top is real and is left alone — recovery reaches 1.0 to
        machine precision around t=0.93 at V=1193, so tau is flat above it and many u map
        to the same t. `sampling_grid` collapses those, which is still the right handling.
        """
        ts = np.linspace(0.0, 1.0, points)
        recovery = recovery_curve(ts, vocab_size, alpha_fn, beta_fn, gamma_fn, mu_accuracy)
        return cls.from_recovery(
            ts, recovery, vocab_size, flat_eps=0.0, p_e0=float(1.0 - recovery[0]),
            source="simulated",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TimeReparametrization":
        """Rebuild a curve written by `save`, exactly, whatever it was built from."""
        data = np.load(path)
        return cls(data["t_knots"], data["u_knots"], source=str(data["source"]))

    def save(self, path: str | Path) -> None:
        """Write the knots, which are the whole map, as `.npz`."""
        np.savez(path, t_knots=self.t_knots, u_knots=self.u_knots, source=self.source)

    # -- the map ----------------------------------------------------------------------

    def t_of(self, u: ArrayLike) -> ArrayLike:
        """tau^-1: the flow time a uniform draw u maps to. Clamps outside [0, 1].

        A tensor is interpolated on its own device. Training calls this once per step, so
        routing it through numpy would mean a device-to-host copy — and therefore a full
        CUDA sync, blocking on every kernel still in flight — inside the hot loop.
        """
        if self.is_identity:
            return u
        return self._interp(u, self.u_knots, self.t_knots, 0)

    def u_of(self, t: ArrayLike) -> ArrayLike:
        """tau: the uniform draw that lands at flow time t. 0 below t_start, 1 above t_end.

        Training never needs this direction — it draws u first. It is for code that picks
        t first, such as a probe at fixed flow times.
        """
        if self.is_identity:
            return t
        return self._interp(t, self.t_knots, self.u_knots, 1)

    # -- read-outs --------------------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """Whether this is no reparametrization at all (`schedule.enabled: false`)."""
        return self.source == "identity"

    @property
    def t_start(self) -> float:
        """tau^-1(0): where the draw, and sampling, begin — the edge of the dead zone."""
        return float(self.t_of(0.0))

    @property
    def t_end(self) -> float:
        """tau^-1(1): where the draw ends — where recovery saturates."""
        return float(self.t_of(1.0))

    @property
    def median_t(self) -> float:
        """The t half a uniform batch lands at or below. tau^-1 is t's quantile function,
        so this is tau^-1(0.5) exactly, with no sampling and no seed to wobble."""
        return float(self.t_of(0.5))

    def quantiles(self, qs: ArrayLike) -> np.ndarray:
        """The flow times at quantiles `qs` in [0, 1] of the training draw."""
        return np.asarray(self.t_of(np.asarray(qs, dtype=np.float64)), dtype=np.float64)

    def sampling_grid(self, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """(t_n, u_n) with u_n = n/N and t_n = tau^-1(u_n) — the sampler's schedule.

        Training and generation have to share this. Once the model is trained under the
        reparameterization it has seen almost nothing below the dead zone, so integrating
        on a uniform-in-t grid afterwards would run it far out of distribution. u_n comes
        back too, for a denoiser fed `flow.time_input: uniform`.

        Repeated t are dropped, keeping the first u of each run, so `num_steps` is a budget
        rather than a promise. Wherever tau is flat, several u map to the same t, and the
        integrator would re-run the denoiser on a state it has already seen — measured on a
        trained checkpoint, five of sixteen steps all landed on t=1.000 and integrated
        nothing. Collapsing them skips the dead interval and spends the steps that remain
        on t that actually differ.

        This is `bin_edges(num_steps)` up to that collapse: the two share a construction,
        but a bin may legitimately be empty while a step may not be a no-op.
        """
        u = np.linspace(0.0, 1.0, num_steps + 1)
        t = np.asarray(self.t_of(u), dtype=np.float64)
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, u = t[keep], u[keep]
        if t.size < 2:  # a constant tau leaves nothing to integrate along
            return np.array([float(t[0]) if t.size else 0.0, 1.0]), np.array([0.0, 1.0])
        return t, u

    def bin_edges(self, num_bins: int) -> torch.Tensor:
        """Bin edges at the k/n quantiles of the training draw, so no bin sits empty.

        Equal-width bins are the wrong grid under a reparametrization — it puts nothing
        below the dead zone, so ~65% of them would stay empty and every sample would crowd
        into the top two or three. Under the identity these are equal-width anyway.
        """
        if self.is_identity:
            return torch.linspace(0, 1, num_bins + 1)
        edges = self.quantiles(np.linspace(0.0, 1.0, num_bins + 1))
        if np.all(np.diff(edges) > 0):
            return torch.as_tensor(edges, dtype=torch.float32)

        # A measured tau saturates: once the denoiser recovers everything above some t, tau
        # is 1 there and tau^-1 is flat, so the top quantiles land on the same t. Those bins
        # would hold almost nothing anyway, so nudge them apart rather than refusing to bin
        # at all — raising here would kill a run for a schedule that is perfectly usable.
        span = float(edges[-1] - edges[0])
        if span <= 0:
            raise ValueError(
                f"{self!r} is constant on [0, 1] at t={edges[0]:.4f}, so there is no "
                f"schedule to bin along. The model recovers nothing anywhere; train further "
                f"first."
            )
        eps = span * 1e-6
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        return torch.as_tensor(edges, dtype=torch.float32)

    def __repr__(self) -> str:
        """`TimeReparametrization(measured, t in [0.512, 0.997], median 0.803, 143 knots)`."""
        return (
            f"TimeReparametrization({self.source}, t in [{self.t_start:.3f}, "
            f"{self.t_end:.3f}], median {self.median_t:.3f}, {self.t_knots.size} knots)"
        )

    # -- internals --------------------------------------------------------------------

    def _interp(self, x: ArrayLike, xk: np.ndarray, yk: np.ndarray, direction: int):
        """Piecewise-linear y(x) through (xk, yk), matching np.interp's clamp at the ends."""
        if not torch.is_tensor(x):
            out = np.interp(np.asarray(x, dtype=np.float64), xk, yk)
            return float(out) if np.ndim(x) == 0 else out
        if xk.size == 1:  # the curve never rises: every input lands on the one knot
            return torch.full_like(x, float(yk[0]))

        key = (x.device, direction)
        if key not in self._device_knots:
            # float64 so the interpolation matches np.interp to ~1e-9 rather than the
            # ~4e-5 float32 accumulates; it is a handful of values, so the cost is nil.
            self._device_knots[key] = (
                torch.as_tensor(xk, dtype=torch.float64, device=x.device),
                torch.as_tensor(yk, dtype=torch.float64, device=x.device),
            )
        xk_t, yk_t = self._device_knots[key]
        # Below the first knot w goes negative and above the last it exceeds 1, and the
        # clamp pins both to the endpoint values.
        xd = x.detach().double()
        idx = torch.searchsorted(xk_t, xd.contiguous(), right=True).clamp(1, xk_t.numel() - 1)
        lo, hi = xk_t[idx - 1], xk_t[idx]
        w = ((xd - lo) / (hi - lo).clamp_min(torch.finfo(torch.float64).tiny)).clamp(0, 1)
        return torch.lerp(yk_t[idx - 1], yk_t[idx], w).to(x.dtype)
