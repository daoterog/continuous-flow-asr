"""Centered isotonic regression: the monotone curve a noisy sweep is evidence for.

tau has to increase for tau^-1 to exist, but a measured recovery curve does not: it is a
finite sample, and the denoiser it reads need not improve at every t. Something has to
impose the monotonicity, and which something is not a detail.

`np.maximum.accumulate` — a running max — was doing it. It only ever ratchets up, so an
upward noise spike propagates to every later t and is never averaged back down. Two costs,
both measured (scripts/schedule/ablate_tau_estimator.py):

  * it is biased upward, and the bias GROWS with resolution, because more bins means more
    draws for the running max to find. Over 50 -> 400 bins its error quadruples while the
    isotonic fit is flat. A resolution knob that makes the answer worse is not a knob.
  * on a curve with a genuine dip it does not converge to its own definition. Its noise-free
    limit puts t0 at 0.497; from 2048 utterances it reports 0.110, because one spike inside
    the dip breaks the flat run early.

Pool-adjacent-violators is the alternative, and `sklearn.isotonic` is the implementation:
the monotone least-squares fit, which for binomial proportions weighted by their trial
counts is also the maximum-likelihood one. It averages violators instead of stepping over
them.

What sklearn does not do is the centring. Plain PAVA returns a step function and is known
to spike at the boundaries, fitting the extreme observations rather than shrinking them —
which matters here because tau^-1(0) is exactly a boundary value and it decides where the
sampler starts. Centring (Oron & Flournoy) collapses each block to its weighted centroid in
x, which both smooths the steps and pulls the ends in. Measured against a known truth it
roughly halves the error at t0.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def centered_isotonic(
    x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The monotone fit of y on x, as (x, y) knots — one per pooled block, at its centroid.

    `w` weights each observation; for a recovery measured over `n` tokens that is `n`, which
    makes the fit the binomial MLE. Equal weights if omitted.

    `x` must be sorted ascending, and at least one weight must be non-zero — a fit with
    nothing behind it is a caller's bug, and sklearn raises on it anyway, so this says so
    in terms of the sweep rather than of sample weights.

    The result increases in y: PAVA gives one value per block and the blocks are what we
    return, so no two consecutive knots carry the same value. On a perfectly flat run the
    rise can be a single ulp — PAVA's own merged means drift in the last bit — which
    `TimeReparametrization.from_recovery` absorbs, since it interpolates with the gap clamped away from zero.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x and y must be matching 1-D arrays, got {x.shape} and {y.shape}")
    if x.size == 0:
        return x, y
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    if w.shape != y.shape:
        raise ValueError(f"w must match y, got {w.shape} and {y.shape}")
    if np.any(np.diff(x) < 0):
        raise ValueError("x must be sorted ascending")
    if w.sum() <= 0:
        raise ValueError(
            "every observation has zero weight, so there is nothing to fit; a sweep that "
            "found no real tokens should be caught before it gets here"
        )

    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
        x, y, sample_weight=w
    )
    # PAVA writes one value across each pooled block, so blocks are runs of exactly equal
    # floats and an equality test finds them without a tolerance.
    ends = np.append(np.flatnonzero(np.diff(fitted) != 0), fitted.size - 1)
    starts = np.concatenate([[0], ends[:-1] + 1])

    centres = np.empty(ends.size, dtype=np.float64)
    for k, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        block = slice(lo, hi + 1)
        # A zero-weight block carries no evidence about where in x it sits, so fall back to
        # the plain midpoint rather than dividing by nothing.
        centres[k] = (
            np.average(x[block], weights=w[block])
            if w[block].sum() > 0
            else x[block].mean()
        )
    return centres, fitted[ends]
