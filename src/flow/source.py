"""The source distribution: x0 ~ N(mu, sigma^2), and where sigma comes from.

The path is

    x_t = alpha(t) (sigma * x0) + beta(t) x1 + gamma(t) mu,     x0 ~ N(0, I)

with alpha(t) = 1 - t a pure shape, and sigma owned here. Two kinds of source:

  * fixed    sigma is a float, `flow.alpha.scale` (1 when unset). Every run before
             `flow.learned_noise` is this one.
  * learned  each canvas position picks one of K preset scales (`learned_noise.scales`)
             from logits off mu's network: a Gumbel-softmax sample in training (hard
             forward, the softmax's gradient backward), the most likely preset at
             inference. sigma is (B, L, 1), one value per position, broadcast over the
             vocabulary. A variance-only KL (`kl`) holds the choice to `target_scale`.

Presets rather than a continuous scale: every position's sigma is then exactly one of K
values, so its recovery curve is exactly a row of a pre-built table (`src.flow.tau_table`),
and each utterance's time curve is exact rather than read off an averaged sigma.

Everything that draws the noise — training's interpolant, both samplers, the audio-only
probe — takes sigma from here and multiplies it into the time coefficient before the draw,
`(alpha(t) * sigma) * x0`, so a fixed sigma gives bit-for-bit the numbers it gave when it
was folded into alpha. Code that reads the path's noise coefficient as one function of t
(the recovery curve, the simulated tau) gets it from `coefficient`.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.flow.coefficients import as_spec, make_schedule

DEFAULT_SCALES = (0.1, 0.2, 0.4, 0.7, 1.0)


class SourceNoise:
    """sigma for a run: a fixed float, or one learned preset per canvas position."""

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
            self.scales = (fixed_sigma,)
            self.init_index = 0
            return
        self.scales = tuple(float(s) for s in (spec.get("scales") or DEFAULT_SCALES))
        self.sigma0 = float(spec.get("target_scale", 0.4))
        if list(self.scales) != sorted(set(self.scales)) or self.scales[0] <= 0.0:
            raise ValueError(f"flow.learned_noise.scales must be positive and increasing, got "
                             f"{list(self.scales)}")
        # Matched to a relative 1e-6: a grid written from a formula (2^(k/4) steps) carries
        # its values rounded, and 0.4 must still find 0.4.
        near = [i for i, s in enumerate(self.scales) if abs(s - self.sigma0) <= 1e-6 * self.sigma0]
        if not near:
            raise ValueError(
                f"flow.learned_noise.target_scale {self.sigma0} must be one of the scales "
                f"{list(self.scales)}: the run starts on it"
            )
        self.init_index = near[0]
        self.sigma0 = self.scales[self.init_index]
        self.kl_weight = float(spec.get("kl_weight", 0.1))
        self.temperature = float(spec.get("temperature", 1.0))
        ratio2 = torch.tensor(self.scales).div(self.sigma0).square()
        # The variance-only KL(N(mu, s_k^2) || N(mu, sigma_0^2)) of each preset.
        self._preset_kl = 0.5 * (ratio2 - 1.0 - ratio2.log())

    @property
    def num_presets(self) -> int:
        """K, the presets a position chooses between; 1 for a fixed source."""
        return len(self.scales)

    def sigma(self) -> float:
        """sigma_0: the fixed source's sigma, or the learned source's target."""
        return self.sigma0

    def choose(
        self, logits: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """((B, L, 1) sigma, (B, L, K) one-hot choice) from the head's (B, L, K) logits.

        Training samples the choice by Gumbel-softmax with `hard=True`: the forward pass
        sees one preset, the backward pass the softmax's gradient. Inference takes the most
        likely preset, so a sampled transcript does not depend on a coin.
        """
        logits = logits.float()
        if training:
            onehot = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        else:
            onehot = F.one_hot(logits.argmax(dim=-1), self.num_presets).float()
        scales = torch.tensor(self.scales, device=logits.device)
        return (onehot * scales).sum(dim=-1, keepdim=True), onehot

    def kl(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """The expected preset KL under the head's softmax, averaged over real positions.

        Differentiable through the probabilities, and bounded below by the smallest
        preset, which is as far as the choice can collapse.
        """
        p = logits.float().softmax(dim=-1)
        per_pos = (p * self._preset_kl.to(p.device)).sum(dim=-1)
        mask = real.float()
        return (per_pos * mask).sum() / mask.sum().clamp_min(1.0)

    def coefficient(self, sigma: float | None = None) -> Callable:
        """The noise coefficient as one function of t, sigma * alpha(t), at a scalar sigma.

        For code that takes the whole coefficient — `TimeReparametrization.exact` and the
        recovery curve behind it. Defaults to sigma_0.
        """
        sigma = self.sigma0 if sigma is None else float(sigma)
        return lambda t: sigma * self.shape(t)
