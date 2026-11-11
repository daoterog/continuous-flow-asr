"""The source distribution: x0 ~ N(mu, sigma^2), and where sigma comes from.

The path is

    x_t = alpha(t) (sigma * x0) + beta(t) x1 + gamma(t) mu,     x0 ~ N(0, I)

with alpha(t) = 1 - t a pure shape, and sigma owned here. Two kinds of source:

  * fixed    sigma is a float, `flow.alpha.scale` (1 when unset). Every run before
             `flow.learned_noise` is this one.
  * learned  sigma is a (B, L, V) tensor read off mu's network,
             sigma = sigma_0 (m + (1 - m) exp(z / 2)), m = min_scale / sigma_0, so it starts
             at exactly sigma_0 when the head's raw output z is 0, never falls to min_scale,
             and is held to sigma_0 by a variance-only KL (`kl`).

Everything that draws the noise — training's interpolant, both samplers, the audio-only
probe — takes sigma from here and multiplies it into the time coefficient before the draw,
`(alpha(t) * sigma) * x0`, so a fixed sigma gives bit-for-bit the numbers it gave when it
was folded into alpha. Code that reads the path's noise coefficient as one function of t
(the recovery curve, the simulated tau) gets it from `coefficient`.
"""

from collections.abc import Callable

import torch
from omegaconf import DictConfig

from src.flow.coefficients import as_spec, make_schedule


class SourceNoise:
    """sigma for a run: a fixed float, or a learned per-coordinate tensor with its KL."""

    def __init__(self, flow: DictConfig | dict) -> None:
        """Read the source from the `flow` node; see the module docstring for the two kinds."""
        # `flow.alpha.scale` is a fixed source's sigma; what is left is alpha's time shape.
        alpha = as_spec(flow.get("alpha", "linear_down"))
        fixed_sigma = float(alpha.pop("scale", 1.0))
        self.shape = make_schedule(alpha)

        spec = flow.get("learned_noise")
        self.learned = bool(spec)
        if not self.learned:
            self.sigma0 = fixed_sigma
            return
        self.sigma0 = float(spec.get("target_scale", 1.0))
        self.kl_weight = float(spec.get("kl_weight", 0.1))
        self.min_scale = float(spec.get("min_scale", 0.05))
        if not 0.0 <= self.min_scale < self.sigma0:
            raise ValueError(
                f"flow.learned_noise.min_scale must be in [0, target_scale), got "
                f"{self.min_scale} against {self.sigma0}"
            )

    def sigma(self, z: torch.Tensor | None = None) -> float | torch.Tensor:
        """The source's sigma: the fixed float, or the learned tensor from the head's `z`."""
        if not self.learned:
            return self.sigma0
        m = self.min_scale / self.sigma0
        return self.sigma0 * (m + (1.0 - m) * torch.exp(0.5 * z.float().clamp(max=4.0)))

    def kl(self, sigma: torch.Tensor) -> torch.Tensor:
        """KL(N(mu, sigma^2) || N(mu, sigma_0^2)), averaged per coordinate.

        Variance only: both share mu, so the mean is left free to move toward the target.
        """
        ratio2 = (sigma / self.sigma0).square()
        return 0.5 * (ratio2 - 1.0 - torch.log(ratio2)).mean()

    def coefficient(self, sigma: float | None = None) -> Callable:
        """The noise coefficient as one function of t, sigma * alpha(t), at a scalar sigma.

        For code that takes the whole coefficient — `TimeReparametrization.exact` and the
        recovery curve behind it. Defaults to sigma_0.
        """
        sigma = self.sigma0 if sigma is None else float(sigma)
        return lambda t: sigma * self.shape(t)
