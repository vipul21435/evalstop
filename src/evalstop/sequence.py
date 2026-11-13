"""Result container and shared machinery for the confidence sequences.

Every method works on the unit scale internally: scores in ``[lo, hi]`` are
mapped to ``[0, 1]`` on the way in and the limits are mapped back on the way
out.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

# Rounding slack allowed when checking that a score lies inside its bounds.
_SLACK = 1e-12


@dataclass(frozen=True)
class Stop:
    """The look at which a stopping rule fired.

    ``n`` is the number of scores seen at that look. ``decision`` is ``"above"``
    or ``"below"`` for a threshold rule and ``"width"`` for a width rule.
    ``lower`` and ``upper`` are the limits at that look.
    """

    n: int
    decision: str
    lower: float
    upper: float


@dataclass(frozen=True, eq=False)
class ConfidenceSequence:
    """Confidence limits for a mean after each score.

    ``lower[t - 1]`` and ``upper[t - 1]`` are the limits after ``t`` scores and
    ``mean[t - 1]`` is the running mean. If the scores share one conditional
    mean, that mean lies inside every interval at once with probability at
    least ``1 - alpha``, so the sequence may be inspected after every score and
    a decision taken at any look.
    """

    lower: FloatArray
    upper: FloatArray
    mean: FloatArray
    alpha: float
    method: str
    bounds: tuple[float, float]

    def __len__(self) -> int:
        return int(self.lower.shape[0])

    @property
    def width(self) -> FloatArray:
        """Width of the interval after each score (zero where it is empty)."""
        return np.maximum(self.upper - self.lower, 0.0)

    @property
    def first_empty(self) -> int | None:
        """Number of scores after which the interval first became empty, if it did.

        A running intersection can only become empty when the scores do not
        share one mean (for example the order is sorted by difficulty) or on
        the rare event the sequence is allowed to miss.
        """
        hits = np.flatnonzero(self.lower > self.upper)
        return int(hits[0]) + 1 if hits.size else None

    def threshold_stop(self, threshold: float) -> Stop | None:
        """Return the first look at which ``threshold`` is outside the interval.

        The decision is ``"above"`` when the lower limit has reached the
        threshold (the mean is above it) and ``"below"`` when the upper limit
        has. ``None`` means the comparison was still open after the last score.
        """
        lo, hi = self.bounds
        if not lo < threshold < hi:
            raise ValueError(f"threshold must lie strictly between {lo:g} and {hi:g}")
        # Limits on a rescaled grid can be off by a rounding error.
        slack = _SLACK * (hi - lo)
        above = self.lower >= threshold - slack
        hits = np.flatnonzero(above | (self.upper <= threshold + slack))
        if hits.size == 0:
            return None
        i = int(hits[0])
        decision = "above" if above[i] else "below"
        return Stop(i + 1, decision, float(self.lower[i]), float(self.upper[i]))

    def inside_stop(self, low: float, high: float) -> Stop | None:
        """Return the first look at which the whole interval lies in ``[low, high]``.

        Used as a futility rule: once the interval for a difference lies inside
        ``(-delta, delta)``, no difference of at least ``delta`` is left to find.
        """
        if not low < high:
            raise ValueError("low must be below high")
        slack = _SLACK * (self.bounds[1] - self.bounds[0])
        ok = (
            (self.lower >= low - slack)
            & (self.upper <= high + slack)
            & (self.lower <= self.upper)
        )
        hits = np.flatnonzero(ok)
        if hits.size == 0:
            return None
        i = int(hits[0])
        return Stop(i + 1, "inside", float(self.lower[i]), float(self.upper[i]))

    def width_stop(self, width: float) -> Stop | None:
        """Return the first look at which the interval is no wider than ``width``."""
        if not width > 0:
            raise ValueError("width must be positive")
        ok = (self.upper - self.lower <= width) & (self.lower <= self.upper)
        hits = np.flatnonzero(ok)
        if hits.size == 0:
            return None
        i = int(hits[0])
        return Stop(i + 1, "width", float(self.lower[i]), float(self.upper[i]))


def check_alpha(alpha: float) -> float:
    """Return ``alpha`` as a float after checking that it lies in (0, 1)."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return alpha


def check_bounds(lo: float, hi: float) -> tuple[float, float]:
    """Return ``(lo, hi)`` as floats after checking that they are finite and ordered."""
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("score bounds must be finite with lo < hi")
    return lo, hi


def unit_scale(scores: ArrayLike | Iterable[float], lo: float, hi: float) -> FloatArray:
    """Validate ``scores`` and map them from ``[lo, hi]`` to ``[0, 1]``."""
    if not isinstance(scores, (np.ndarray, list, tuple)):
        scores = list(scores)
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite numbers")
    unit = (arr - lo) / (hi - lo)
    if unit.size and (unit.min() < -_SLACK or unit.max() > 1.0 + _SLACK):
        raise ValueError(f"scores must lie in [{lo:g}, {hi:g}]")
    return np.clip(unit, 0.0, 1.0)


class OnlineCS(ABC):
    """A confidence sequence that is updated one score at a time.

    Subclasses implement :meth:`_step`, which receives the new score on the
    unit scale and returns the interval for the current look, also on the unit
    scale. This class handles validation, rescaling and the running
    intersection.
    """

    method: str = ""

    def __init__(self, alpha: float, lo: float, hi: float, intersect: bool) -> None:
        self.alpha = check_alpha(alpha)
        self.lo, self.hi = check_bounds(lo, hi)
        self.intersect = bool(intersect)
        self._n = 0
        self._sum = 0.0
        self._lower = 0.0
        self._upper = 1.0

    @abstractmethod
    def _step(self, unit_score: float) -> tuple[float, float]:
        """Take one unit-scale score and return this look's unit-scale interval.

        Called before ``self._n`` and ``self._sum`` are updated.
        """

    @property
    def n(self) -> int:
        """Number of scores seen so far."""
        return self._n

    @property
    def mean(self) -> float:
        """Running mean of the scores (``nan`` before the first one)."""
        if self._n == 0:
            return math.nan
        return self.lo + (self.hi - self.lo) * (self._sum / self._n)

    @property
    def lower(self) -> float:
        """Current lower limit."""
        return self.lo + (self.hi - self.lo) * self._lower

    @property
    def upper(self) -> float:
        """Current upper limit."""
        return self.lo + (self.hi - self.lo) * self._upper

    @property
    def interval(self) -> tuple[float, float]:
        """Current ``(lower, upper)`` limits."""
        return self.lower, self.upper

    def update(self, score: float) -> tuple[float, float]:
        """Add one score and return the new ``(lower, upper)`` limits."""
        try:
            value = float(score)
        except (TypeError, ValueError):
            raise ValueError(f"score {score!r} is not a number") from None
        if not math.isfinite(value):
            raise ValueError("scores must be finite numbers")
        unit = (value - self.lo) / (self.hi - self.lo)
        if unit < -_SLACK or unit > 1.0 + _SLACK:
            raise ValueError(f"score {value!r} is outside [{self.lo:g}, {self.hi:g}]")
        unit = min(max(unit, 0.0), 1.0)
        lower, upper = self._step(unit)
        self._n += 1
        self._sum += unit
        if self.intersect:
            self._lower = max(self._lower, lower)
            self._upper = min(self._upper, upper)
        else:
            self._lower, self._upper = lower, upper
        return self.interval

    def run(self, scores: Iterable[float]) -> ConfidenceSequence:
        """Add every score in order and return the limits after each one."""
        lower: list[float] = []
        upper: list[float] = []
        mean: list[float] = []
        for score in scores:
            lo, hi = self.update(score)
            lower.append(lo)
            upper.append(hi)
            mean.append(self.mean)
        return ConfidenceSequence(
            lower=np.asarray(lower, dtype=float),
            upper=np.asarray(upper, dtype=float),
            mean=np.asarray(mean, dtype=float),
            alpha=self.alpha,
            method=self.method,
            bounds=(self.lo, self.hi),
        )
