"""Normal-mixture confidence sequence for the mean of bounded scores.

A score in [0, 1] minus its conditional mean is sub-Gaussian with variance
proxy 1/4 (Hoeffding's lemma). With ``S_t`` the centred sum and ``V_t = t / 4``,
the process ``exp(lam * S_t - lam**2 * V_t / 2)`` is therefore a nonnegative
supermartingale for every real ``lam``. Averaging it over ``lam ~ N(0, 1 / rho)``
keeps that property and has the closed form

    sqrt(rho / (V_t + rho)) * exp(S_t**2 / (2 * (V_t + rho))).

Ville's inequality bounds the probability that this ever reaches ``1 / alpha``
by ``alpha``, and reaching it is the event

    |S_t| >= sqrt((V_t + rho) * log((V_t + rho) / (rho * alpha**2))).

The right-hand side is the two-sided normal-mixture boundary, eq. (14) of
Howard, Ramdas, McAuliffe and Sekhon, "Time-uniform, nonparametric,
nonasymptotic confidence sequences", Annals of Statistics 49(2), 2021. Their
Proposition 3 gives the value of ``rho`` that makes the boundary tightest at a
chosen time.

The variance proxy 1/4 is the worst case for scores in [0, 1], so this
sequence does not get narrower when the pass rate is close to 0 or 1. The
betting sequence in :mod:`evalstop.betting` does.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from evalstop.sequence import (
    ConfidenceSequence,
    FloatArray,
    OnlineCS,
    check_alpha,
    check_bounds,
    unit_scale,
)

# Variance proxy of one score on the unit scale (Hoeffding's lemma).
_UNIT_VARIANCE_PROXY = 0.25


def normal_mixture_boundary(v: ArrayLike, alpha: float, rho: float) -> FloatArray:
    """Two-sided normal-mixture boundary at intrinsic time ``v``.

    For a process ``S_t`` that is sub-Gaussian with variance process ``V_t``,
    the probability that ``|S_t| >= normal_mixture_boundary(V_t, alpha, rho)``
    at any time is at most ``alpha``.
    """
    alpha = check_alpha(alpha)
    if not rho > 0:
        raise ValueError("rho must be positive")
    v = np.asarray(v, dtype=float)
    return np.sqrt((v + rho) * np.log((v + rho) / (rho * alpha * alpha)))


def best_rho(v_opt: float, alpha: float) -> float:
    """Value of ``rho`` that makes the boundary smallest at intrinsic time ``v_opt``.

    Setting the derivative of the squared boundary with respect to ``rho`` to
    zero gives ``y - log(y) = 1 + 2 * log(1 / alpha)`` with ``y = 1 + v_opt / rho``.
    The root above 1 equals ``-W(-alpha**2 / e)`` on the lower branch of the
    Lambert W function; it is found here by Newton's method.
    """
    alpha = check_alpha(alpha)
    if not v_opt > 0:
        raise ValueError("v_opt must be positive")
    target = 1.0 + 2.0 * math.log(1.0 / alpha)
    y = target + math.log(target)
    for _ in range(100):
        step = (y - math.log(y) - target) / (1.0 - 1.0 / y)
        y -= step
        if abs(step) <= 1e-15 * y:
            break
    return v_opt / (y - 1.0)


def _check_tune_at(tune_at: float) -> float:
    tune_at = float(tune_at)
    if not (math.isfinite(tune_at) and tune_at > 0):
        raise ValueError("tune_at must be a positive number of scores")
    return tune_at


class MixtureCS(OnlineCS):
    """Normal-mixture confidence sequence, updated one score at a time.

    Parameters
    ----------
    alpha:
        The sequence covers the mean at every look with probability at least
        ``1 - alpha``.
    lo, hi:
        Bounds of a score.
    tune_at:
        Number of scores at which the interval should be tightest. The
        sequence is valid for any value; it is somewhat wider far away from
        this point.
    intersect:
        Report the intersection of all intervals so far (the default), which
        makes the limits monotone.
    """

    method = "mixture"

    def __init__(
        self,
        alpha: float = 0.05,
        *,
        lo: float = 0.0,
        hi: float = 1.0,
        tune_at: float = 200,
        intersect: bool = True,
    ) -> None:
        super().__init__(alpha, lo, hi, intersect)
        self.tune_at = _check_tune_at(tune_at)
        self._rho = best_rho(self.tune_at * _UNIT_VARIANCE_PROXY, self.alpha)

    def _step(self, unit_score: float) -> tuple[float, float]:
        t = self._n + 1
        mean = (self._sum + unit_score) / t
        radius = float(normal_mixture_boundary(t * _UNIT_VARIANCE_PROXY, self.alpha, self._rho)) / t
        return max(mean - radius, 0.0), min(mean + radius, 1.0)


def mixture_cs(
    scores: ArrayLike | Iterable[float],
    alpha: float = 0.05,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    tune_at: float = 200,
    intersect: bool = True,
) -> ConfidenceSequence:
    """Normal-mixture confidence sequence for the mean of ``scores``.

    Same result as feeding the scores to :class:`MixtureCS` one by one, computed
    in one vectorised pass.
    """
    alpha = check_alpha(alpha)
    lo, hi = check_bounds(lo, hi)
    tune_at = _check_tune_at(tune_at)
    unit = unit_scale(scores, lo, hi)
    t = np.arange(1, unit.size + 1, dtype=float)
    mean = np.cumsum(unit) / t
    rho = best_rho(tune_at * _UNIT_VARIANCE_PROXY, alpha)
    radius = normal_mixture_boundary(t * _UNIT_VARIANCE_PROXY, alpha, rho) / t
    lower = np.maximum(mean - radius, 0.0)
    upper = np.minimum(mean + radius, 1.0)
    if intersect:
        lower = np.maximum.accumulate(lower)
        upper = np.minimum.accumulate(upper)
    span = hi - lo
    return ConfidenceSequence(
        lower=lo + span * lower,
        upper=lo + span * upper,
        mean=lo + span * mean,
        alpha=alpha,
        method=MixtureCS.method,
        bounds=(lo, hi),
    )
