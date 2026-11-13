"""Stopping rules for a paired comparison, vectorised over many runs.

Every rule takes ``d``, an array of shape ``(runs, items)`` with the paired
score differences (model A minus model B) in evaluation order, and returns for
each run the number of items used and the decision. A one-dimensional ``d`` is
one run.

Rules that stay valid however often they are checked:

* ``betting``: the hedged-capital test of "the mean difference is 0", which is
  the event that the betting confidence sequence excludes 0;
* ``mixture``: the same with the normal-mixture confidence sequence;
* ``discordant``: a mixture likelihood-ratio test on the discordant pairs, the
  sequential analogue of McNemar's test (binary scores).

Baselines:

* ``naive``: the exact McNemar test repeated at every look, stopping the first
  time it is significant. Its error rate is not ``alpha``;
* ``fixed``: the exact McNemar test once, after the last item.

Classical sequential designs, valid under their own conditions:

* ``obf`` and ``pocock``: group-sequential McNemar z-tests at a few planned
  looks with alpha-spending boundaries (valid at those looks only, and by a
  normal approximation);
* ``sprt``: two one-sided Wald tests on the discordant pairs, of a win rate of
  1/2 against a chosen win rate in either direction, which can also stop for
  futility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evalstop.gsd import design
from evalstop.mixture import best_rho, normal_mixture_boundary
from evalstop.sequence import check_alpha
from evalstop.sprt import wald_thresholds
from evalstop.stats import discordant_log_evalue, mcnemar_critical

A_BETTER = 1
B_BETTER = -1
UNDECIDED = 0
FUTILE = 2


@dataclass(frozen=True)
class Outcome:
    """Result of one rule on every run.

    ``stop`` is the number of items used (all of them when the rule never
    fired). ``decision`` is ``A_BETTER``, ``B_BETTER``, ``UNDECIDED`` or
    ``FUTILE`` (the difference is smaller than the chosen minimum effect).
    """

    stop: NDArray[np.int64]
    decision: NDArray[np.int8]


def _paths(d: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(d, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("differences must have shape (runs, items) with at least one item")
    if not np.all(np.isfinite(arr)):
        raise ValueError("differences must be finite numbers")
    return arr


def _first_true(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Per row, the 1-based position of the first True, or 0 when there is none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1) + 1, 0)


def _combine(
    n: int,
    up: NDArray[np.int64],
    down: NDArray[np.int64],
    futile: NDArray[np.int64] | None = None,
) -> Outcome:
    """Turn first-crossing times (0 means never) into stop times and decisions."""
    never = n + 1
    up_t = np.where(up > 0, up, never)
    down_t = np.where(down > 0, down, never)
    efficacy = np.minimum(up_t, down_t)
    decided = efficacy < never
    decision = np.where(decided, np.where(up_t <= down_t, A_BETTER, B_BETTER), UNDECIDED)
    stop = np.where(decided, efficacy, n)
    if futile is not None:
        futile_t = np.where(futile > 0, futile, never)
        earlier = futile_t < efficacy
        decision = np.where(earlier, FUTILE, decision)
        stop = np.where(earlier, futile_t, stop)
    return Outcome(stop.astype(np.int64), decision.astype(np.int8))


def _check_effect(min_effect: float | None, scale: float) -> None:
    if min_effect is not None and not 0.0 < min_effect < scale:
        raise ValueError("min_effect must lie strictly between 0 and the largest difference")


def betting_rule(
    d: ArrayLike,
    alpha: float = 0.05,
    *,
    min_effect: float | None = None,
    scale: float = 1.0,
    c: float = 0.5,
) -> Outcome:
    """Hedged-capital test of a zero mean difference, with optional futility.

    Tracks the capital of :class:`evalstop.BettingCS` at the candidate
    difference 0 only, so it stops exactly when that confidence sequence
    excludes 0. With ``min_effect`` it also tracks the candidates
    ``-min_effect`` and ``+min_effect`` and stops for futility once both are
    excluded, that is once the interval lies inside ``(-min_effect, min_effect)``.
    ``scale`` is the largest possible absolute difference.
    """
    alpha = check_alpha(alpha)
    _check_effect(min_effect, scale)
    paths = _paths(d)
    runs, n = paths.shape
    unit = (paths / scale + 1.0) / 2.0
    if unit.min() < -1e-12 or unit.max() > 1.0 + 1e-12:
        raise ValueError("differences must lie between -scale and scale")
    points = [0.5]
    if min_effect is not None:
        points += [(1.0 - min_effect / scale) / 2.0, (1.0 + min_effect / scale) / 2.0]
    m = np.asarray(points)
    cap_plus = c / m
    cap_minus = c / (1.0 - m)
    log_reject = math.log(2.0 / alpha)
    bet_scale = 2.0 * math.log(2.0 / alpha)
    log_plus = np.zeros((runs, m.size))
    log_minus = np.zeros((runs, m.size))
    first_plus = np.zeros((runs, m.size), dtype=np.int64)
    first_minus = np.zeros((runs, m.size), dtype=np.int64)
    total = np.zeros(runs)
    sq_dev = np.zeros(runs)
    for t in range(1, n + 1):
        u = unit[:, t - 1]
        variance = (0.25 + sq_dev) / t
        bet = np.sqrt(bet_scale / (variance * t * math.log(t + 1.0)))[:, None]
        diff = u[:, None] - m[None, :]
        log_plus += np.log1p(np.minimum(bet, cap_plus) * diff)
        log_minus += np.log1p(-np.minimum(bet, cap_minus) * diff)
        total += u
        sq_dev += (u - (0.5 + total) / (t + 1.0)) ** 2
        np.copyto(first_plus, t, where=(first_plus == 0) & (log_plus >= log_reject))
        np.copyto(first_minus, t, where=(first_minus == 0) & (log_minus >= log_reject))
    futile = None
    if min_effect is not None:
        above_low = first_plus[:, 1]
        below_high = first_minus[:, 2]
        futile = np.where((above_low > 0) & (below_high > 0), np.maximum(above_low, below_high), 0)
    return _combine(n, first_plus[:, 0], first_minus[:, 0], futile)


def mixture_rule(
    d: ArrayLike,
    alpha: float = 0.05,
    *,
    tune_at: float = 200,
    min_effect: float | None = None,
    scale: float = 1.0,
) -> Outcome:
    """Normal-mixture test of a zero mean difference, with optional futility.

    Stops exactly when :func:`evalstop.mixture_cs` on the differences excludes
    0, or for futility when it lies inside ``(-min_effect, min_effect)``.
    """
    alpha = check_alpha(alpha)
    _check_effect(min_effect, scale)
    paths = _paths(d)
    n = paths.shape[1]
    t = np.arange(1, n + 1, dtype=float)
    rho = best_rho(tune_at * 0.25, alpha)
    radius = 2.0 * scale * normal_mixture_boundary(t * 0.25, alpha, rho) / t
    mean = np.cumsum(paths, axis=1) / t
    up = _first_true(mean - radius >= 0.0)
    down = _first_true(mean + radius <= 0.0)
    futile = None
    if min_effect is not None:
        above_low = _first_true(mean - radius >= -min_effect)
        below_high = _first_true(mean + radius <= min_effect)
        futile = np.where((above_low > 0) & (below_high > 0), np.maximum(above_low, below_high), 0)
    return _combine(n, up, down, futile)


def _counts(paths: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return np.cumsum(paths > 0, axis=1), np.cumsum(paths < 0, axis=1)


def _directed(
    n: int, first: NDArray[np.int64], wins: NDArray[np.int64], losses: NDArray[np.int64]
) -> Outcome:
    rows = np.arange(first.shape[0])
    at = np.maximum(first - 1, 0)
    a_ahead = wins[rows, at] > losses[rows, at]
    up = np.where((first > 0) & a_ahead, first, 0)
    down = np.where((first > 0) & ~a_ahead, first, 0)
    return _combine(n, up, down)


def discordant_rule(d: ArrayLike, alpha: float = 0.05, *, prior: float = 1.0) -> Outcome:
    """Mixture likelihood-ratio test on discordant pairs (sequential McNemar).

    A pair counts as a win when the difference is positive and as a loss when
    it is negative; ties are ignored. Stops when the mixture likelihood ratio
    of :func:`evalstop.stats.discordant_log_evalue` reaches ``1 / alpha``.
    """
    alpha = check_alpha(alpha)
    paths = _paths(d)
    wins, losses = _counts(paths)
    crossed = discordant_log_evalue(wins, losses, prior) >= math.log(1.0 / alpha)
    return _directed(paths.shape[1], _first_true(crossed), wins, losses)


def _mcnemar_rule(paths: NDArray[np.float64], alpha: float, looks: NDArray[np.bool_]) -> Outcome:
    wins, losses = _counts(paths)
    k = wins + losses
    crit = mcnemar_critical(alpha, int(k.max()))
    significant = (np.minimum(wins, losses) <= crit[k]) & looks[None, :]
    return _directed(paths.shape[1], _first_true(significant), wins, losses)


def naive_rule(d: ArrayLike, alpha: float = 0.05, *, every: int = 1) -> Outcome:
    """Exact McNemar test repeated at every look, stopping at the first p <= alpha.

    ``every`` is the number of items between looks; the last item is always a
    look. This is the baseline that peeking amounts to. It is not valid: its
    false-positive rate grows with the number of looks.
    """
    alpha = check_alpha(alpha)
    if every < 1:
        raise ValueError("every must be at least 1")
    paths = _paths(d)
    looks = np.zeros(paths.shape[1], dtype=bool)
    looks[every - 1 :: every] = True
    looks[-1] = True
    return _mcnemar_rule(paths, alpha, looks)


def fixed_rule(d: ArrayLike, alpha: float = 0.05) -> Outcome:
    """Exact McNemar test once, after the last item (the fixed-sample analysis)."""
    alpha = check_alpha(alpha)
    paths = _paths(d)
    looks = np.zeros(paths.shape[1], dtype=bool)
    looks[-1] = True
    return _mcnemar_rule(paths, alpha, looks)


def look_items(n: int, looks: int) -> list[int]:
    """Item counts of ``looks`` equally spaced looks over ``n`` items."""
    if looks < 1:
        raise ValueError("looks must be at least 1")
    return sorted({math.ceil(n * (j + 1) / looks) for j in range(looks)})


def group_sequential_rule(
    d: ArrayLike, alpha: float = 0.05, *, looks: int = 5, spending: str = "obf"
) -> Outcome:
    """McNemar z-test at equally spaced planned looks with alpha-spending boundaries.

    At each look the statistic ``(wins - losses) / sqrt(wins + losses)`` is
    compared with the boundary of :func:`evalstop.gsd.design` for the share of
    items seen so far.
    """
    alpha = check_alpha(alpha)
    paths = _paths(d)
    n = paths.shape[1]
    at = look_items(n, looks)
    plan = design([t / n for t in at], alpha, spending)
    wins, losses = _counts(paths)
    index = np.asarray(at) - 1
    w, lo = wins[:, index], losses[:, index]
    z = (w - lo) / np.sqrt(np.maximum(w + lo, 1))
    crossed = np.abs(z) >= np.asarray(plan.z)[None, :]
    look = _first_true(crossed)
    first = np.where(look > 0, np.asarray(at)[np.maximum(look - 1, 0)], 0)
    return _directed(n, first, wins, losses)


def sprt_rule(
    d: ArrayLike, alpha: float = 0.05, *, beta: float = 0.2, win_rate: float = 0.65
) -> Outcome:
    """Two one-sided Wald tests on the discordant pairs.

    One tests "a discordant pair favours A with probability 1/2" against
    ``win_rate``, the other against ``1 - win_rate``, each at ``alpha / 2``
    with Wald's thresholds. The run stops for A or B when the matching test
    chooses its alternative, and for futility once both have chosen 1/2.
    """
    alpha = check_alpha(alpha)
    if not 0.5 < win_rate < 1.0:
        raise ValueError("win_rate must lie strictly between 0.5 and 1")
    upper, lower = wald_thresholds(alpha / 2.0, beta)
    paths = _paths(d)
    n = paths.shape[1]
    toward = math.log(win_rate / 0.5)
    away = math.log((1.0 - win_rate) / 0.5)
    win, loss = paths > 0, paths < 0
    llr_a = np.cumsum(win * toward + loss * away, axis=1)
    llr_b = np.cumsum(win * away + loss * toward, axis=1)

    def ends(llr: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        chose_alt, chose_null = _first_true(llr >= upper), _first_true(llr <= lower)
        alt_first = (chose_alt > 0) & ((chose_null == 0) | (chose_alt < chose_null))
        null_first = (chose_null > 0) & ~alt_first
        return np.where(alt_first, chose_alt, 0), np.where(null_first, chose_null, 0)

    a_alt, a_null = ends(llr_a)
    b_alt, b_null = ends(llr_b)
    futile = np.where((a_null > 0) & (b_null > 0), np.maximum(a_null, b_null), 0)
    return _combine(n, a_alt, b_alt, futile)


RULES = ("betting", "mixture", "discordant", "obf", "pocock", "sprt", "naive", "fixed")


def run_rule(
    name: str,
    d: ArrayLike,
    alpha: float = 0.05,
    *,
    min_effect: float | None = None,
    tune_at: float = 200,
    every: int = 1,
    prior: float = 1.0,
    scale: float = 1.0,
    looks: int = 5,
    beta: float = 0.2,
    win_rate: float = 0.65,
) -> Outcome:
    """Run the rule called ``name`` on the difference paths ``d``."""
    if name == "betting":
        return betting_rule(d, alpha, min_effect=min_effect, scale=scale)
    if name == "mixture":
        return mixture_rule(d, alpha, tune_at=tune_at, min_effect=min_effect, scale=scale)
    if name == "discordant":
        return discordant_rule(d, alpha, prior=prior)
    if name in ("obf", "pocock"):
        return group_sequential_rule(d, alpha, looks=looks, spending=name)
    if name == "sprt":
        return sprt_rule(d, alpha, beta=beta, win_rate=win_rate)
    if name == "naive":
        return naive_rule(d, alpha, every=every)
    if name == "fixed":
        return fixed_rule(d, alpha)
    raise ValueError(f"unknown rule {name!r}; expected one of {', '.join(RULES)}")


def summarise(outcome: Outcome, n: int) -> dict[str, float]:
    """Shares of each decision and the number of items used, over the runs."""
    decision = outcome.decision
    return {
        "a_better": float(np.mean(decision == A_BETTER)),
        "b_better": float(np.mean(decision == B_BETTER)),
        "futile": float(np.mean(decision == FUTILE)),
        "undecided": float(np.mean(decision == UNDECIDED)),
        "mean_items": float(np.mean(outcome.stop)),
        "median_items": float(np.median(outcome.stop)),
        "mean_share": float(np.mean(outcome.stop)) / n,
    }
