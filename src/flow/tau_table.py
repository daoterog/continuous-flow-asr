"""A time curve per utterance, read off a table of pre-built recovery curves.

The simulated tau is built at one sigma and one mu accuracy for the whole run. Under a
learned source neither is one number: each position takes its own preset sigma
(`src.flow.source`), recovery is nonlinear in sigma, and the curve at the mean sigma is not
the mean of the curves — under the per-coordinate source it came out a near-step while the
field really decoded over a wide band.

So the averaging moves to the curves. `R[k, a, t]` is the exact field recovery
(`recovery.recovery_curve`) for preset k at mu confidence a, built once. Each canvas
position takes the row of the preset it chose, interpolated in mu's confidence there; an
utterance's curve is the mean over its real positions, normalised to tau exactly as the
simulated curve is, and mixed with the identity at the uniform floor so it starts at t = 0.
Nothing in it needs the target, so sampling computes the same curve training drew from.

A fixed source is one preset, so the same path gives a fixed-sigma run its per-utterance
curve, varying only with mu's confidence.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch

from src.flow.recovery import recovery_curve


class TauTable:
    """`R[preset, mu confidence, t]` and the per-utterance curves read off it."""

    def __init__(
        self,
        vocab_size: int,
        alpha_shape: Callable,
        beta_fn: Callable,
        gamma_fn: Callable | None,
        scales: Sequence[float],
        accuracy_bins: int = 8,
        t_points: int = 128,
        soft_mu: bool = True,
    ) -> None:
        """Build the table: one exact recovery curve per (preset, mu confidence) pair.

        `recovery_curve` models mu as one-hot, right with probability a. The real mu is
        soft: a token it gives probability p adds gamma * p to the field, not gamma. Under
        `soft_mu` the curve at confidence p is the one-hot curve with gamma scaled by p and
        accuracy p — mu's top token carrying its own mass, and being the target as often as
        a calibrated mu says. Measured on the sigma-0.4 control, that tracks the field's real
        recovery about twice as closely as the one-hot reading (`soft_mu=False`) and as the
        old global curve (scripts/schedule/validate_tau_table.py).
        """
        self.scales = tuple(float(s) for s in scales)
        accuracies = np.linspace(0.0, 1.0, accuracy_bins)
        ts = np.linspace(0.0, 1.0, t_points)
        table = np.empty((len(self.scales), accuracy_bins, t_points))
        for i, s in enumerate(self.scales):
            for j, a in enumerate(accuracies):
                gamma = gamma_fn
                if soft_mu and gamma_fn is not None:
                    gamma = lambda t, a=a: a * gamma_fn(t)  # noqa: E731
                table[i, j] = recovery_curve(
                    ts, vocab_size, lambda t, s=s: s * alpha_shape(t), beta_fn, gamma, a
                )
        self.accs = torch.tensor(accuracies, dtype=torch.float32)
        self.ts = torch.tensor(ts, dtype=torch.float32)
        self.table = torch.tensor(table, dtype=torch.float32)

    def to(self, device: torch.device | str) -> "TauTable":
        """Move the table's tensors; returns self."""
        for name in ("accs", "ts", "table"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def recovery_tokens(self, preset: torch.Tensor, a_pos: torch.Tensor) -> torch.Tensor:
        """(B, L, T) field recovery per token: its preset's row at mu's confidence there.

        `preset` is each position's chosen preset index, (B, L); `a_pos` mu's top
        probability there. The preset picks its row exactly; the confidence is linear
        between the table's neighbouring bins.
        """
        a = a_pos.float().clamp(0.0, 1.0)
        j = (torch.searchsorted(self.accs, a.contiguous(), right=True) - 1)
        j = j.clamp(0, self.accs.numel() - 2)
        w = ((a - self.accs[j]) / (self.accs[j + 1] - self.accs[j]))[..., None]
        return (1 - w) * self.table[preset, j] + w * self.table[preset, j + 1]

    def recovery(
        self, preset: torch.Tensor, a_pos: torch.Tensor, real: torch.Tensor
    ) -> torch.Tensor:
        """(B, T) field recovery per utterance: its real positions' curves, averaged."""
        mask = real[..., None].float()
        curves = self.recovery_tokens(preset, a_pos)
        return (curves * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def curves(self, recovery: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
        """The curve u = tau(t) on the table's t grid, from recovery over its last axis.

        Per utterance, (B, T), or per token, (B, L, T). tau = 1 - P_e(t) / P_e(0), as
        `schedule.tau_from_recovery` with the path's own error at t = 0, then the uniform
        floor u' = (1 - w) tau + w t, which also keeps a token mu already has right (P_e(0)
        near 0) on a curve that still reaches t = 1 at u = 1.
        """
        p_e = 1.0 - recovery
        tau = (1.0 - p_e / p_e[..., :1].clamp_min(1e-12)).clamp(0.0, 1.0)
        tau = torch.cummax(tau, dim=-1).values
        return (1.0 - floor) * tau + floor * self.ts

    def invert(self, u_rows: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """The t with tau(t) = u, for each curve in `u_rows` (..., T).

        `u` is one value per curve, shaped like `u_rows` without its last axis, or K per
        curve, (..., K); the result has `u`'s shape.
        """
        lead, n_t = u_rows.shape[:-1], self.ts.numel()
        many = u.dim() == u_rows.dim()
        rows = u_rows.reshape(-1, n_t).contiguous()
        want = (u if many else u[..., None]).reshape(rows.size(0), -1).contiguous()
        hi = torch.searchsorted(rows, want).clamp(1, n_t - 1)
        u0, u1 = rows.gather(1, hi - 1), rows.gather(1, hi)
        w = ((want - u0) / (u1 - u0).clamp_min(1e-12)).clamp(0.0, 1.0)
        t = self.ts[hi - 1] + w * (self.ts[hi] - self.ts[hi - 1])
        return t.reshape(*lead, -1) if many else t.reshape(lead)

    def grid(self, u_rows: torch.Tensor, num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """((..., N+1) flow times, (N+1,) u) for an N-step sampler: equal steps in u.

        One row of times per curve: (B, N+1) for per-utterance curves, (B, L, N+1) for
        per-token ones, every curve reaching the same u at the same step.
        """
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=u_rows.device)
        t = self.invert(u_rows, u.expand(*u_rows.shape[:-1], -1).contiguous())
        t[..., 0], t[..., -1] = 0.0, 1.0
        return t, u
