"""Wald's sequential probability ratio test for a pass rate.

To decide between a pass rate ``p0`` and a pass rate ``p1`` with error rates
``alpha`` (choosing ``p1`` when ``p0`` is true) and ``beta`` (choosing ``p0``
when ``p1`` is true), follow the log-likelihood ratio

    llr_n = S_n * log(p1 / p0) + (n - S_n) * log((1 - p1) / (1 - p0))

with ``S_n`` the number of passes among the first ``n`` results, and stop the
first time it leaves the interval ``(log B, log A)`` with Wald's thresholds

    A = (1 - beta) / alpha,        B = beta / (1 - alpha).

Leaving at the top chooses ``p1``, at the bottom ``p0``. In terms of ``S_n``
the two boundaries are parallel straight lines. With these thresholds the true
error rates satisfy ``alpha' <= alpha / (1 - beta)`` and
``beta' <= beta / (1 - alpha)`` (A. Wald, "Sequential Tests of Statistical
Hypotheses", Annals of Mathematical Statistics 16(2), 1945).

Unlike a confidence sequence the test needs both hypotheses fixed in advance,
and in exchange it can also stop in favour of ``p0``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evalstop.sequence import check_alpha


def wald_thresholds(alpha: float, beta: float) -> tuple[float, float]:
    """Return ``(log A, log B)`` for error rates ``alpha`` and ``beta``."""
    alpha = check_alpha(alpha)
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly between 0 and 1")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be below 1")
    return math.log((1.0 - beta) / alpha), math.log(beta / (1.0 - alpha))


def _check_rates(p0: float, p1: float) -> None:
    if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0) or p0 == p1:
        raise ValueError("p0 and p1 must differ and lie strictly between 0 and 1")


@dataclass(frozen=True)
class SprtResult:
    """Outcome of a test: results used, the choice (``"h1"``, ``"h0"`` or ``None``), the path."""

    n: int
    decision: str | None
    llr: NDArray[np.float64]


class BernoulliSPRT:
    """Wald's test of pass rate ``p0`` against ``p1``, updated one result at a time."""

    def __init__(self, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.2) -> None:
        _check_rates(p0, p1)
        self.p0, self.p1 = float(p0), float(p1)
        self.upper, self.lower = wald_thresholds(alpha, beta)
        self._pass = math.log(p1 / p0)
        self._fail = math.log((1.0 - p1) / (1.0 - p0))
        self.llr = 0.0
        self.n = 0
        self.decision: str | None = None

    def update(self, passed: bool) -> str | None:
        """Add one result; return ``"h1"`` or ``"h0"`` once the test has stopped."""
        if self.decision is None:
            self.n += 1
            self.llr += self._pass if passed else self._fail
            if self.llr >= self.upper:
                self.decision = "h1"
            elif self.llr <= self.lower:
                self.decision = "h0"
        return self.decision

    def lines(self) -> tuple[float, float, float]:
        """Return ``(slope, h1, h0)`` of the boundaries in terms of the pass count.

        For ``p1 > p0`` the test chooses ``p1`` when ``S_n >= h1 + slope * n``
        and ``p0`` when ``S_n <= h0 + slope * n``; for ``p1 < p0`` both
        inequalities are reversed.
        """
        scale = self._pass - self._fail
        return -self._fail / scale, self.upper / scale, self.lower / scale


def sprt_bernoulli(
    results: Iterable[float], p0: float, p1: float, alpha: float = 0.05, beta: float = 0.2
) -> SprtResult:
    """Run Wald's test over pass/fail ``results`` and stop at its first decision."""
    test = BernoulliSPRT(p0, p1, alpha, beta)
    path = []
    for value in results:
        test.update(bool(value))
        path.append(test.llr)
        if test.decision is not None:
            break
    return SprtResult(test.n, test.decision, np.asarray(path, dtype=float))


def expected_sample_size(
    p: float, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.2
) -> float:
    """Wald's approximation to the mean number of results when the pass rate is ``p0`` or ``p1``.

    ``p`` must equal one of the two hypotheses. The approximation ignores the
    overshoot of the last step, so the true mean is a little larger.
    """
    _check_rates(p0, p1)
    upper, lower = wald_thresholds(alpha, beta)
    if p == p0:
        mean_end = (1.0 - alpha) * lower + alpha * upper
    elif p == p1:
        mean_end = beta * lower + (1.0 - beta) * upper
    else:
        raise ValueError("p must equal p0 or p1")
    drift = p * math.log(p1 / p0) + (1.0 - p) * math.log((1.0 - p1) / (1.0 - p0))
    return mean_end / drift
