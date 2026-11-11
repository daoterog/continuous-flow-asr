"""Token recovery under the interpolant, in closed form.

``schedule.sweep_corruption`` answers the same question by Monte Carlo: how often does
``argmax I_t`` land on the right token, with no denoiser involved. That probability is a
Gaussian order statistic, so it can be integrated instead of sampled.

The field at one canvas position, with mu one-hot and correct with probability `a`:

    I_t = alpha(t) z + beta(t) e_L + gamma(t) mu,      z ~ N(0, I_V)

Divide through by alpha(t) and the slots are V shifted unit Gaussians. Two cases:

  * mu right (probability a) — the spike and mu stack on slot L, V-1 rivals sit at 0:

        q_right = E_z[ Phi(z + (beta + gamma)/alpha)^(V-1) ]

  * mu wrong (probability 1-a) — slot L is at beta/alpha, mu puts ONE rival at
    gamma/alpha, and V-2 rivals remain at 0:

        q_wrong = E_z[ Phi(z + (beta - gamma)/alpha) * Phi(z + beta/alpha)^(V-2) ]

  * recovery = a * q_right + (1 - a) * q_wrong

With gamma absent the second case collapses into the first and both reduce to the
one-term form, which is FMLM's ``compute_alpha_exact`` (see src/models/fmlm/schedule.py).
Checked against ``sweep_corruption`` for all three variants at several mu accuracies: the
largest disagreement in tau was 0.007 against a Monte Carlo noise floor of ~0.006 at 24k
canvas positions, i.e. they agree as closely as the sweep can be measured.

The expectations are Gauss-Hermite quadrature in log space, the same construction FMLM
uses: `log_ndtr` for a numerically safe `log Phi`, products become sums, and one `exp`
at the end.

Two things the exactness does NOT buy, both worth knowing before trusting the curve:

  * It is strictly increasing in exact arithmetic but NOT in float64. Recovery reaches
    1.0 to machine precision around t = 0.93 at V = 1193 and is flat from there up, so
    tau saturates and many u map to the same t. That is a true statement about the path
    rather than an artefact — P_e is ~1e-17 there, and there really is nothing left to
    decode — so `TimeReparametrization.sampling_grid`'s collapse of repeated t is still needed, and still
    right. What the exact curve does remove is the LOW end: there is no flat run below
    the dead zone, so `tau^-1(0)` lands where the curve says instead of wherever
    `flat_eps` and the keep-last-of-each-flat-run rule happened to leave it.
  * `tau_from_recovery`'s default `p_e0 = 1 - 1/V` is the chance error rate, which is
    only the path's error at t=0 when gamma(0) == 0. For `audio_cond` gamma(0) == 1, so
    the field already carries mu at t=0 and recovery starts above chance: tau(0) comes out
    at 0.012 instead of 0 at mu accuracy 0.9, and the offset grows as mu learns. The
    normaliser wants to be `1 - recovery[0]`, the error where the path actually begins.
"""

from collections.abc import Callable

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import log_ndtr

# Quadrature nodes. FMLM's default, and the curve is flat in it well below this.
N_NODES = 100

# alpha(t) -> 0 as t -> 1, and it divides every shift. Floored rather than special-cased:
# the shifts go to +inf, Phi saturates at 1 and recovery is 1, which is the right answer.
ALPHA_FLOOR = 1e-12


def _nodes(n_gh: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights rescaled for an expectation under N(0, 1).

    `hermgauss` integrates against exp(-x^2); the standard normal needs x -> sqrt(2) x
    and weights divided by sqrt(pi).
    """
    x, w = hermgauss(n_gh)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def recovery_curve(
    ts: np.ndarray | list[float],
    vocab_size: int,
    alpha_fn: Callable,
    beta_fn: Callable,
    gamma_fn: Callable | None = None,
    mu_accuracy: float = 0.0,
    n_gh: int = N_NODES,
) -> np.ndarray:
    """P[argmax I_t == target], one value per t. The exact form of `sweep_corruption`.

    Takes the same coefficient callables the run trains with, so the curve describes the
    probability path the model actually sees. `mu_accuracy` is how often the stand-in mu is
    right, matching `simulate_mu`; it is ignored when `gamma_fn` is None.

    Vectorised over `ts` — the cost is one (len(ts), n_gh) array, so a 10,000-point grid is
    milliseconds and `num_bins` stops being a precision tradeoff.
    """
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be at least 2, got {vocab_size}")
    if not 0.0 <= mu_accuracy <= 1.0:
        raise ValueError(f"mu_accuracy must be in [0, 1], got {mu_accuracy}")

    ts = np.asarray(ts, dtype=np.float64)
    z, w = _nodes(n_gh)
    z = z[None, :]

    alpha = np.maximum(np.asarray(alpha_fn(ts), dtype=np.float64), ALPHA_FLOOR)[:, None]
    beta = np.asarray(beta_fn(ts), dtype=np.float64)[:, None]
    m_target = beta / alpha

    if gamma_fn is None:
        return np.sum(w * np.exp((vocab_size - 1) * log_ndtr(z + m_target)), axis=-1)

    gamma = np.asarray(gamma_fn(ts), dtype=np.float64)[:, None]
    m_mu = gamma / alpha

    # mu agrees with the target: one spike of beta + gamma against V-1 untouched slots.
    log_right = (vocab_size - 1) * log_ndtr(z + m_target + m_mu)
    q_right = np.sum(w * np.exp(log_right), axis=-1)

    # mu disagrees: the spike must also outrun the slot mu backed. Broadcasting both
    # shifts over the nodes keeps this a single quadrature rather than one per t.
    log_wrong = log_ndtr(z + m_target - m_mu) + (vocab_size - 2) * log_ndtr(z + m_target)
    q_wrong = np.sum(w * np.exp(log_wrong), axis=-1)

    return mu_accuracy * q_right + (1.0 - mu_accuracy) * q_wrong
