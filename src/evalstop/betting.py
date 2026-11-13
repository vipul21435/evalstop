"""Hedged-capital (betting) confidence sequence for the mean of bounded scores.

For a candidate mean ``m`` two wealth processes start at 1 and bet on each
score ``x`` in [0, 1]:

    K_plus(m)  *= 1 + lam_plus(m)  * (x - m)     (grows when the mean is above m)
    K_minus(m) *= 1 - lam_minus(m) * (x - m)     (grows when the mean is below m)

The bets are fixed before the score is seen and capped so that the wealth
stays positive: ``lam_plus(m) <= c / m`` and ``lam_minus(m) <= c / (1 - m)``
with ``c < 1``. If ``m`` is the conditional mean of every score, both processes
are nonnegative martingales, and so is ``theta * K_plus + (1 - theta) * K_minus``.
By Ville's inequality that sum reaches ``1 / alpha`` with probability at most
``alpha``, and the hedged capital ``max(theta * K_plus, (1 - theta) * K_minus)``
is never larger. The confidence set after ``t`` scores is every ``m`` whose
hedged capital is still below ``1 / alpha``.

This is the hedged capital process and Theorem 3 of Waudby-Smith and Ramdas,
"Estimating means of bounded random variables by betting", Journal of the
Royal Statistical Society Series B 86(1), 2024 (eqs. 24 and 25 of the arXiv
version, 2010.09686). The size of the bet is their predictable plug-in
(eq. 26),

    lam_t = sqrt(2 * log(2 / alpha) / (var_{t-1} * t * log(t + 1))),

with ``var_t = (1/4 + sum_{i<=t} (x_i - mu_i)**2) / (t + 1)`` and
``mu_t = (1/2 + sum_{i<=t} x_i) / (t + 1)``. It uses only the scores already
seen and adapts to their variance.

``K_plus`` is nonincreasing in ``m`` and ``K_minus`` is nondecreasing, so the
set is an interval. It is tracked on an evenly spaced grid of candidate means.
Each reported limit is the nearest rejected grid point outside the set (or the
score bound when no grid point on that side is rejected), so the grid can make
the interval wider but never narrower.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from evalstop.sequence import ConfidenceSequence, OnlineCS


class BettingCS(OnlineCS):
    """Hedged-capital confidence sequence, updated one score at a time.

    Parameters
    ----------
    alpha:
        The sequence covers the mean at every look with probability at least
        ``1 - alpha``.
    lo, hi:
        Bounds of a score.
    grid:
        Number of cells in the grid of candidate means. Each update costs time
        proportional to it, and the limits are multiples of ``(hi - lo) / grid``.
    c:
        Cap on the bets, strictly between 0 and 1.
    theta:
        Share of the starting wealth given to the bet that the mean is above
        the candidate; the rest bets that it is below.
    intersect:
        Report the intersection of all intervals so far (the default), which
        makes the limits monotone.
    """

    method = "betting"

    def __init__(
        self,
        alpha: float = 0.05,
        *,
        lo: float = 0.0,
        hi: float = 1.0,
        grid: int = 1000,
        c: float = 0.5,
        theta: float = 0.5,
        intersect: bool = True,
    ) -> None:
        super().__init__(alpha, lo, hi, intersect)
        if int(grid) != grid or grid < 2:
            raise ValueError("grid must be a whole number of at least 2 cells")
        if not 0.0 < c < 1.0:
            raise ValueError("c must lie strictly between 0 and 1")
        if not 0.0 < theta < 1.0:
            raise ValueError("theta must lie strictly between 0 and 1")
        self.grid = int(grid)
        self.c = float(c)
        self.theta = float(theta)
        self._m = np.arange(self.grid + 1, dtype=float) / self.grid
        with np.errstate(divide="ignore"):
            self._cap_plus = self.c / self._m
            self._cap_minus = self.c / (1.0 - self._m)
        self._log_plus = np.zeros(self.grid + 1)
        self._log_minus = np.zeros(self.grid + 1)
        self._log_theta = math.log(self.theta)
        self._log_one_minus_theta = math.log1p(-self.theta)
        self._log_reject = -math.log(self.alpha)
        self._bet_scale = 2.0 * math.log(2.0 / self.alpha)
        self._sq_dev = 0.0

    def _step(self, unit_score: float) -> tuple[float, float]:
        t = self._n + 1
        variance = (0.25 + self._sq_dev) / t
        bet = math.sqrt(self._bet_scale / (variance * t * math.log(t + 1.0)))
        diff = unit_score - self._m
        self._log_plus += np.log1p(np.minimum(bet, self._cap_plus) * diff)
        self._log_minus += np.log1p(-np.minimum(bet, self._cap_minus) * diff)
        mu_hat = (0.5 + self._sum + unit_score) / (t + 1.0)
        self._sq_dev += (unit_score - mu_hat) ** 2

        log_capital = np.maximum(
            self._log_theta + self._log_plus,
            self._log_one_minus_theta + self._log_minus,
        )
        inside = np.flatnonzero(log_capital < self._log_reject)
        if inside.size:
            first, last = int(inside[0]), int(inside[-1])
        else:
            # Every grid point is rejected: the set lies inside the two cells
            # around the point with the smallest capital.
            first = last = int(np.argmin(log_capital))
        lower = self._m[max(first - 1, 0)]
        upper = self._m[min(last + 1, self.grid)]
        return float(lower), float(upper)


def betting_cs(
    scores: Iterable[float],
    alpha: float = 0.05,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    grid: int = 1000,
    c: float = 0.5,
    theta: float = 0.5,
    intersect: bool = True,
) -> ConfidenceSequence:
    """Hedged-capital confidence sequence for the mean of ``scores``.

    Feeds the scores to :class:`BettingCS` in order and returns the limits
    after each one.
    """
    monitor = BettingCS(alpha, lo=lo, hi=hi, grid=grid, c=c, theta=theta, intersect=intersect)
    return monitor.run(scores)
