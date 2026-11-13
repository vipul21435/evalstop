"""Small numerical helpers, written with numpy and the standard library only."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

_NORMAL = NormalDist()
_erfc = np.frompyfunc(math.erfc, 1, 1)


def norm_cdf(x: ArrayLike) -> NDArray[np.float64]:
    """Standard normal distribution function, elementwise."""
    values = np.asarray(x, dtype=float)
    return 0.5 * np.asarray(_erfc(-values / math.sqrt(2.0)), dtype=float)


def norm_ppf(p: float) -> float:
    """Standard normal quantile."""
    return _NORMAL.inv_cdf(p)


def mcnemar_exact_p(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for ``wins`` and ``losses`` discordant pairs.

    Under the null each discordant pair favours either model with probability
    1/2, so the smaller count is compared with a Binomial(wins + losses, 1/2)
    lower tail, doubled and capped at 1.
    """
    if wins < 0 or losses < 0:
        raise ValueError("counts must not be negative")
    k = wins + losses
    if k == 0:
        return 1.0
    low = min(wins, losses)
    log_half = -k * math.log(2.0)
    tail = sum(
        math.exp(math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(k - j + 1) + log_half)
        for j in range(low + 1)
    )
    return min(1.0, 2.0 * tail)


def mcnemar_critical(alpha: float, kmax: int) -> NDArray[np.int64]:
    """Critical values of the exact McNemar test for 0..kmax discordant pairs.

    Entry ``k`` is the largest ``c`` such that a smaller count of ``c`` among
    ``k`` discordant pairs gives a two-sided p-value of at most ``alpha``, or
    -1 when no outcome with ``k`` pairs is significant.
    """
    crit = np.full(kmax + 1, -1, dtype=np.int64)
    pmf = np.zeros(kmax + 2)
    pmf[0] = 1.0
    for k in range(1, kmax + 1):
        pmf[1 : k + 1] = 0.5 * (pmf[1 : k + 1] + pmf[0:k])
        pmf[0] *= 0.5
        significant = np.flatnonzero(2.0 * np.cumsum(pmf[: k // 2 + 1]) <= alpha)
        if significant.size:
            crit[k] = significant[-1]
    return crit


def log_evalue_tables(prior: float, kmax: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tables of ``lgamma(prior + j)`` and ``lgamma(2 * prior + j)`` for j = 0..kmax."""
    one = np.array([math.lgamma(prior + j) for j in range(kmax + 1)])
    two = np.array([math.lgamma(2.0 * prior + j) for j in range(kmax + 1)])
    return one, two


def discordant_log_evalue(wins: ArrayLike, losses: ArrayLike, prior: float = 1.0) -> NDArray:
    """Log of the mixture likelihood ratio against "each discordant pair is a fair coin".

    With ``w`` wins and ``l`` losses the likelihood ratio of win probability
    ``p`` against 1/2 is ``(2p)**w * (2(1-p))**l``. Averaged over a
    Beta(prior, prior) distribution for ``p`` it equals

        2**(w + l) * B(prior + w, prior + l) / B(prior, prior).

    Under the null this is a nonnegative martingale that starts at 1.
    """
    if not prior > 0:
        raise ValueError("prior must be positive")
    w = np.asarray(wins, dtype=np.int64)
    lo = np.asarray(losses, dtype=np.int64)
    kmax = int(np.max(w + lo, initial=0))
    one, two = log_evalue_tables(prior, kmax)
    log_norm = 2.0 * math.lgamma(prior) - math.lgamma(2.0 * prior)
    return (w + lo) * math.log(2.0) + one[w] + one[lo] - two[w + lo] - log_norm
