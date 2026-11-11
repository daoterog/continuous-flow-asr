"""FMLM's time schedule, extracted from their utils.py and algo.py so it reads in one go.

Vendored reference, not used by our code. Everything below is copied verbatim from the
FMLM release (`utils.py` for Parts 1-3, `algo.py` for Part 4); only the ordering, the
section banners and the comments marked `# NOTE (ours)` are mine. Nothing here is called
from `src/` — it exists to be read against `src/flow/schedule.py`, and to be run as a
numerical reference for ours.

------------------------------------------------------------------------------------------
Their names vs ours
------------------------------------------------------------------------------------------
Their variable names are the single biggest obstacle to reading this, because `alpha` and
`gamma` both mean something else in our code:

    theirs      ours        what it is
    --------    --------    ------------------------------------------------------------
    gamma       t           continuous flow time in [0, 1]; the weight on the one-hot
                            target in `x_t = (1-t) z + t x_1`. So their `gamma` IS our
                            `beta_fn(t) = t`, and their `1 - gamma` is our `alpha_fn(t)`.
    alpha       tau         decoding progress in [0, 1]: the chance-corrected probability
                            that `argmax x_t` recovers the token.
    lut_g2a     tau         t -> tau, as a cubic spline
    lut_a2g     tau_inverse tau -> t, as a cubic spline
    tau_t       u           the UNIFORM draw. Careful: their `tau_t` is the sampled
                            variable and their `t` is the derived one, which is the same
                            direction as ours (`u ~ U[0,1]`, `t = tau_inverse(u)`), but
                            their `_sample_t_interval` is named for the wrong one.
    t_min/t_max             clamps on the uniform draw, applied in TAU space, not t space.

------------------------------------------------------------------------------------------
The one result that matters: their tau is our tau, in closed form
------------------------------------------------------------------------------------------
`compute_alpha_exact` computes analytically what our `sweep_corruption` estimates by Monte
Carlo. For `x_t = t * e_label + (1-t) * z` with `z ~ N(0, I_V)`:

    argmax is correct  <=>  t + (1-t) z_label > (1-t) z_k  for all k != label
                       <=>  z_label + t/(1-t) > z_k

so with `m_c = t / (1-t)` (their `m_c = gamma / sigma`, `sigma = 1 - gamma`),

    q_c = P[correct] = E_z[ Phi(z + m_c)^(V-1) ]

evaluated by Gauss-Hermite in log space. Then their last line:

    alpha = V/(V-1) * (q_c - 1/V)

And our definition, with `P_e = 1 - q_c` and `P_e(0) = 1 - 1/V`:

    tau = 1 - P_e(t)/P_e(0) = 1 - (1 - q_c)/((V-1)/V) = (V q_c - 1)/(V - 1)
        = V/(V-1) * (q_c - 1/V)

Identical. Our `tau_from_recovery` and their `compute_alpha_exact` are the same quantity;
the only difference is that they integrate it and we sample it. For the linear variant
(gamma_fn is None) we do not need the corruption sweep at all.

------------------------------------------------------------------------------------------
Reading order
------------------------------------------------------------------------------------------
Part 1  the analytic curve t -> tau          `compute_alpha_exact`
Part 2  the splines and the two conversions  `build_luts`, `alpha_to_gamma`, `gamma_to_alpha`
Part 3  the same integral with absorbing states (masked diffusion, unused by FLM)
Part 4  how algo.py consumes it: training draw, corruption, loss, Euler sampler

------------------------------------------------------------------------------------------
Gaps in the paste
------------------------------------------------------------------------------------------
`algo.py:1155` calls `utils.d_alpha_by_d_gamma(t, self.lut_g2a)`, which is NOT in the
utils.py we have. It is `dtau/dt`, used only by the distillation losses (LSD/ESD
reweighting), never by the flow-matching stage. With the splines in hand it is a one-liner
(`lut_g2a.derivative()(t)`), so it is a missing paste rather than a missing idea.
"""

from typing import Union

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr  # stable log


# ==========================================================================================
# PART 1 -- the analytic curve: continuous time (their gamma, our t) -> progress (tau)
# ==========================================================================================

def compute_alpha_exact(gamma: np.ndarray, K: int, n_gh: int = 100, sigma_floor: float = 1e-12, is_diffusion=False) -> np.ndarray:
    """
    Computes q_c (Alpha) from Gamma using Gauss-Hermite integration.
    This is the ground-truth function mapping Gamma -> Alpha.
    """
    gamma = np.asarray(gamma)

    # 1. Standardized means (assuming tau=0, b=1.0 for this conversion)
    sigma = 1.0 - gamma
    if is_diffusion:
        sigma = np.sqrt(sigma)
    sigma = np.maximum(sigma, sigma_floor)

    m_c = gamma / sigma

    # 2. GH nodes/weights
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    # 3. Broadcasting
    m_c_expanded = m_c[:, None]   # (B, 1)
    z_expanded = z_nodes[None, :] # (1, n_gh)

    # 4. Compute Log-CDFs
    # L_cu = log(Phi(z + m_c))
    L_cu = log_ndtr(z_expanded + m_c_expanded)

    # 5. Weighted sum
    # log_prod_c = (K - 1) * L_cu
    log_prod_c = (K - 1) * L_cu
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)

    # Debugged. should consider prob. from uniform noise.
    # NOTE (ours): this line is exactly our tau = 1 - P_e(t)/P_e(0). See the module
    # docstring. q_c is P[argmax correct]; the rescaling removes the 1/K floor.
    alpha = K/(K-1.) * (q_c - 1./K)

    alpha += (gamma-1) * 1e-10 # minor trick to ensure monotonicity

    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha


def compute_alpha_exact_torch(gamma, K: int, x_np, w_np, sigma_floor: float = 1e-12, is_diffusion=False, device=None) -> torch.Tensor:
    """
    Computes q_c (Alpha) from Gamma using Gauss-Hermite integration (PyTorch version).
    """

    dtype = gamma.dtype
    device = gamma.device

    sigma = 1.0 - gamma
    if is_diffusion:
        sigma = torch.sqrt(sigma)

    sigma = torch.maximum(sigma, torch.tensor(sigma_floor, device=device, dtype=dtype))

    m_c = gamma / sigma

    x = torch.tensor(x_np, dtype=dtype, device=device)
    w = torch.tensor(w_np, dtype=dtype, device=device)

    w = w / np.sqrt(np.pi)
    z_nodes = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device)) * x

    m_c_expanded = m_c.unsqueeze(-1)
    z_expanded = z_nodes.unsqueeze(0)

    L_cu = torch.special.log_ndtr(z_expanded + m_c_expanded)

    log_prod_c = (K - 1) * L_cu

    q_c = torch.sum(w.unsqueeze(0) * torch.exp(log_prod_c), dim=-1)

    alpha = (K / (K - 1.0)) * (q_c - (1.0 / K))

    alpha = alpha + (gamma - 1) * 1e-10

    alpha = torch.clamp(alpha, 0.0, 1.0)

    return alpha


# ==========================================================================================
# PART 2 -- the splines, and the two conversions the trainer actually calls
# ==========================================================================================
#
# NOTE (ours): this is their `make_tau_inverse`. Differences worth noting against ours:
#   * they build BOTH directions; we only ever build the inverse
#   * a CubicSpline, not np.interp, so it is C2 but can overshoot (hence the clips in
#     alpha_to_gamma / gamma_to_alpha below)
#   * no dead-zone handling and no flat_eps: the analytic curve is strictly increasing by
#     construction (that is what the `(gamma-1) * 1e-10` nudge above is for), so the
#     duplicate-knot problem we hit never arises for them
#   * `n_points=10000` on a uniform t grid -- 67x our 150 bins, affordable because it is
#     one quadrature, not a forward pass

def build_luts(K: int, n_points: int = 10000, is_diffusion=False) -> tuple[CubicSpline, CubicSpline]:
    """
    Builds two lookup tables (Splines):
    1. Alpha -> Gamma (Forward)
    2. Gamma -> Alpha (Inverse)

    Reverted to Linear (Uniform) spacing.
    Chebyshev nodes concentrate points at 0 and 1, but for large K, the curve
    is often sigmoid-like (flat at ends, steep in middle).
    Uniform spacing captures the transition region better.
    """
    # 1. Create Alpha grid using Uniform Spacing
    # Simple linspace covers the whole range evenly.
    gamma_vals = np.linspace(0.0, 1.0, n_points) # cont.

    # 2. Compute corresponding Gamma grid (Exact)
    alpha_vals = compute_alpha_exact(gamma_vals, K=K, is_diffusion=is_diffusion) # disc.

    # 3. Build Forward Spline (Alpha -> Gamma)
    # Alpha is strictly increasing. Safe.
    lut_g2a = CubicSpline(gamma_vals, alpha_vals)

    # 4. Build Inverse Spline (Gamma -> Alpha)
    # Gamma values must be strictly increasing to be 'x' in CubicSpline.

    # Sort just in case (though usually monotonic)
    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]

    # Remove duplicates in Gamma
    # Duplicates often happen at very low alpha (gamma ~ 1/K) or very high alpha (gamma ~ 1.0)
    # NOTE (ours): np.unique collapses flat runs to their FIRST index, so their tau^-1(0)
    # lands at the LEFT edge of the dead zone. Ours keeps the last, i.e. the right edge.
    # That is a real behavioural difference, not a style one -- see Finding 2 in the notes.
    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]

    # Create Spline
    lut_a2g = CubicSpline(unique_alpha, unique_gamma)

    return lut_a2g, lut_g2a


def alpha_to_gamma(alpha: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Alpha -> Gamma using the LUT.
    """
    # NOTE (ours): this is tau^-1, i.e. what our `tau_inverse` returns. Note the
    # round-trip to numpy on every call -- a device sync per training step, which is
    # exactly what our torch-native searchsorted path avoids.
    if isinstance(alpha, torch.Tensor):
        dtype = alpha.dtype
        gamma = np.clip(lut(alpha.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(gamma).to(alpha.device, dtype=dtype)
    else:
        return np.clip(lut(alpha), 0.0, 1.0)


def gamma_to_alpha(gamma: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Gamma -> Alpha using the LUT.
    """
    # Clip result to [0, 1] to avoid spline overshoot
    if isinstance(gamma, torch.Tensor):
        dtype = gamma.dtype
        alpha = np.clip(lut(gamma.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(alpha).to(gamma.device, dtype=dtype)
    else:
        return np.clip(lut(gamma), 0.0, 1.0)


# ==========================================================================================
# PART 3 -- the same integral with absorbing (mask) states
# ==========================================================================================
#
# NOTE (ours): FLM never calls these -- `build_luts` goes straight to
# `compute_alpha_exact`. They are the masked-diffusion generalisation: K data classes
# competing against M absorbing ones, with a shift `tau` and a noise scale `b`. Included
# because it shows the shape the integral takes when a third term enters the field, which
# is the position our `three_term` variant is in: there, mu adds a V-wide bias that the
# Gauss-Hermite argument above cannot absorb into a single m_c.

def standardized_means(alpha: float, tau: float, b: float, diffusion=False):
    """
    Returns (m_c, m_u, m_a, sigma), where
      sigma = b * (1 - alpha),
      m_c = (alpha - tau) / sigma   (label / 'correct'),
      m_u = -tau / sigma            (other data),
      m_a = 0.0                     (absorbing)
    """
    sigma = b * (1.0 - alpha)
    if diffusion:
      sigma = sigma ** 0.5
    if sigma <= 0.0:
        sigma = 1e-12
    m_c = (alpha - tau) / sigma
    m_u = (-tau) / sigma
    m_a = 0.0
    return m_c, m_u, m_a, sigma


def compute_qs_fast(alpha: float, tau: float, b: float, K: int, M: int, *,
                    n_gh: int = 100, sigma_floor: float = 1e-12,
                    diffusion=False) -> tuple[float, float, float]:
    """
    Returns (q_c, q_u, q_a) using log-stabilized Gauss-Hermite and
    only a constant number of log_ndtr calls per evaluation.

    q_c : probability the label ('correct') class wins (per label)
    q_u : probability a particular non-label data class wins (per class)
    q_a : probability a particular absorbing class wins (per class)
    """
    # standardized means
    m_c, m_u, m_a, sigma = standardized_means(alpha, tau, b, diffusion)
    if sigma < sigma_floor:
        sigma = sigma_floor  # keep GH numerically sane; values remain consistent

    # GH nodes/weights for exp(-x^2); normalize to N(0,1)
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x  # Z ~ N(0,1) evaluated at sqrt(2) x_l

    # --- Precompute the LOG-CDFs for the few unique shifts we need ---
    # 0-shift (same-class competitors)
    L0     = log_ndtr(z_nodes)                # log Phi(z)
    # label vs absorbing / absorbing vs label
    L_ca   = log_ndtr(z_nodes + m_c)          # log Phi(z + (m_c - 0))
    L_ac   = log_ndtr(z_nodes - m_c)          # log Phi(z + (0   - m_c))
    # data vs absorbing / absorbing vs data
    L_ua   = log_ndtr(z_nodes + m_u)          # log Phi(z + (m_u - 0))
    L_au   = log_ndtr(z_nodes - m_u)          # log Phi(z + (0   - m_u))
    # label vs data / data vs label
    d_cu   = m_c - m_u
    L_cu   = log_ndtr(z_nodes + d_cu)         # log Phi(z + (m_c - m_u))
    L_uc   = log_ndtr(z_nodes - d_cu)         # log Phi(z + (m_u - m_c))

    # --- Build node-wise log-products for each grouped case ---
    # Label winner: (K-1) non-label data + M absorbing competitors
    #   log_prod_c(u) = (K-1)*log Phi(z + (m_c - m_u)) + M*log Phi(z + (m_c - 0))
    log_prod_c = (K - 1) * L_cu + M * L_ca

    # Wrong-data winner (per class): 1 label + (K-2) other data + M absorbing
    #   log_prod_u(u) = log Phi(z + (m_u - m_c)) + (K-2)*log Phi(z) + M*log Phi(z + (m_u - 0))
    if K > 1:
        log_prod_u = L_uc + max(K - 2, 0) * L0 + M * L_ua
    else:
        log_prod_u = None  # no wrong-data class exists

    # Absorbing winner (per class): 1 label + (K-1) data + (M-1) absorbing
    #   log_prod_a(u) = log Phi(z + (0 - m_c)) + (K-1)*log Phi(z + (0 - m_u)) + (M-1)*log Phi(z)
    if M > 0:
        log_prod_a = L_ac + (K - 1) * L_au + max(M - 1, 0) * L0
    else:
        log_prod_a = None  # no absorbing class exists

    # --- Weighted sum over nodes; clip exponents for safety ---
    def weighted_exp_sum(logv):
        # return float(np.sum(w * np.exp(np.clip(logv, -745.0, 745.0))))  # -745 ~ float64 underflow
        return float(np.sum(w * np.exp(logv)))  # -745 ~ float64 underflow

    q_c = weighted_exp_sum(log_prod_c)
    q_u = weighted_exp_sum(log_prod_u) if (log_prod_u is not None and K > 1) else 0.0
    q_a = weighted_exp_sum(log_prod_a) if (log_prod_a is not None and M > 0) else 0.0

    return q_c, q_u, q_a


# ==========================================================================================
# PART 4 -- how algo.py consumes it (transcribed, reference only)
# ==========================================================================================
#
# These are methods lifted out of `FLMBase` and `FLM` in algo.py, verbatim, with their
# original line numbers. They are NOT runnable here (no self, no backbone); they are the
# call chain, in the order it runs, so the schedule can be read end to end.


class _FLMScheduleReference:
    """algo.py, transcribed. Do not instantiate."""

    # algo.py:789-791 -- FLMBase.__init__
    def __init__(self, config, tokenizer):
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        # NOTE (ours): built ONCE, at construction, from the vocab size alone. There is no
        # refresh and no model in the loop -- the whole question we have been fighting over
        # for a week does not exist for them, because the analytic curve is exact.
        self.lut_a2g, self.lut_g2a = build_luts(K=self.vocab_size)

    # algo.py:857-877 -- the training draw. Note it samples in TAU space and the clamps
    # t_min/t_max are tau clamps despite the name.
    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        # NOTE (ours): antithetic/stratified sampling -- one draw per batch slot, jittered
        # inside its own 1/n stratum, then permuted. We draw plain uniforms. This is a
        # variance reduction we do not have and could take directly.
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]
        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t = t[:batch_dim]
        return t

    # algo.py:879-885
    def _tau_to_t(self, tau):
        """Convert t to reparameterized time tau."""
        return alpha_to_gamma(tau, self.lut_a2g)

    def _t_to_tau(self, t):
        """Convert t to reparameterized time tau."""
        return gamma_to_alpha(t, self.lut_g2a)

    # algo.py:887-893 -- our `make_interpolant`, two-term only (no mu, no audio branch)
    def corrupt_continuous(self, x0, t):
        """Corrupt data x0 at time t using linear interpolation with Gaussian noise."""
        t = t.unsqueeze(-1).unsqueeze(-1)
        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data

    # algo.py:1034-1051 -- FLM.loss, the whole flow-matching training step
    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t, output_tokens
        B = x0.shape[0]
        tau_t = self._sample_t_interval(B, current_accumulation_step,
                                    t_min=self.t_min, t_max=self.t_max)
        t = self._tau_to_t(tau_t)
        x_t, target_data = self.corrupt_continuous(x0, t)
        # NOTE (ours): THE conditioning difference. They build the state at t but tell the
        # model tau_t, the uniform draw. We pass t. Under a reparameterization that
        # compresses most of [0,1] into a narrow band, tau is the better-spread input.
        f = self.forward(x_t, tau_t) #condition on tau_t
        loss = -(target_data * f).sum(dim=-1)
        self.log('loss', loss.mean(), prog_bar=True)
        if self.config.algo.learnable_loss_weighting is True:
            loss_weight = self.backbone.learnable_loss_weighting(tau_t)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
            self.log('loss_weighted', loss.mean(), prog_bar=True)
        return loss

    # algo.py:1053-1083 -- FLM.generate_samples, the Euler sampler
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples using Euler ODE solver."""
        if num_steps is None:
            num_steps = self.config.sampling.steps
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        # NOTE (ours): uniform in TAU, mapped through tau^-1 -- our `generation_grid`.
        # Note they do NOT dedupe repeated t; the analytic curve is strictly increasing so
        # there is nothing to dedupe.
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)

        for i in range(num_steps):
            tau_t_curr = tau_vals[i]
            tau_t_next = tau_vals[i + 1]
            tau_t_in = tau_t_curr.expand(B)
            t_in = self._tau_to_t(tau_t_in)
            dt = self._tau_to_t(tau_t_next.expand(B)) - t_in
            x_1_pred = self.forward(z, tau_t_in)
            x_1_pred_probs = x_1_pred.exp()

            # NOTE (ours): the last step returns the prediction directly instead of taking
            # an Euler step into t=1, where (1-t) is zero. Our `sample_tokens_euler`
            # relies on the eps clamp instead.
            if i == num_steps - 1:
                z = x_1_pred_probs
                break

            v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
            z = z + dt.view(-1, 1, 1) * v

        return z.argmax(dim=-1)
