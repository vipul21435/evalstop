"""Hedged-capital confidence sequence: direct recomputation, simulation, properties."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalstop import BettingCS, betting_cs, mixture_cs


def reference_capital(
    xs, m: float, alpha: float, c: float = 0.5, theta: float = 0.5
) -> float:
    """Hedged capital at candidate mean ``m``, written straight from the formulas.

    Plain floats and running products, no grid and no logarithms.
    """
    k_plus = k_minus = 1.0
    total = 0.0
    sq_dev = 0.0
    for t, x in enumerate(xs, start=1):
        variance = (0.25 + sq_dev) / t
        bet = math.sqrt(2.0 * math.log(2.0 / alpha) / (variance * t * math.log(t + 1)))
        lam_plus = min(bet, c / m) if m > 0 else bet
        lam_minus = min(bet, c / (1 - m)) if m < 1 else bet
        k_plus *= 1 + lam_plus * (x - m)
        k_minus *= 1 - lam_minus * (x - m)
        total += x
        mu_hat = (0.5 + total) / (t + 1)
        sq_dev += (x - mu_hat) ** 2
    return max(theta * k_plus, (1 - theta) * k_minus)


def test_reference_reproduces_hand_worked_first_steps():
    # alpha = 0.05, one score equal to 1. The first bet is
    # sqrt(2 log(40) / (0.25 * log 2)) = 6.52...
    bet = math.sqrt(2 * math.log(40) / (0.25 * math.log(2)))
    assert bet == pytest.approx(6.52498, abs=1e-5)
    # m = 0.25: both bets are capped (0.5 / 0.25 = 2 and 0.5 / 0.75 = 2/3), so
    # K_plus = 1 + 2 * 0.75 = 2.5, K_minus = 1 - (2/3) * 0.75 = 0.5.
    assert reference_capital([1.0], 0.25, 0.05) == pytest.approx(0.5 * 2.5)
    # m = 0.05: the upward bet is not capped (0.5 / 0.05 = 10 > 6.52).
    assert reference_capital([1.0], 0.05, 0.05) == pytest.approx(0.5 * (1 + bet * 0.95))
    # m = 1 after a score of 0: the downward bet is not capped, K_minus = 1 + bet.
    assert reference_capital([0.0], 1.0, 0.05) == pytest.approx(0.5 * (1 + bet))


@pytest.mark.parametrize(("seed", "alpha", "grid"), [(7, 0.2, 50), (8, 0.05, 40)])
def test_interval_is_the_accepted_grid_points_widened_by_one_cell(seed, alpha, grid):
    rng = np.random.default_rng(seed)
    xs = rng.beta(2.0, 3.0, size=80)
    cs = BettingCS(alpha, grid=grid, intersect=False)
    for t, x in enumerate(xs, start=1):
        lower, upper = cs.update(x)
        accepted = [
            j / grid
            for j in range(grid + 1)
            if reference_capital(xs[:t], j / grid, alpha) < 1.0 / alpha
        ]
        assert accepted, "the grid is fine enough for this check"
        assert lower == pytest.approx(max(min(accepted) - 1.0 / grid, 0.0), abs=1e-12)
        assert upper == pytest.approx(min(max(accepted) + 1.0 / grid, 1.0), abs=1e-12)


def test_cap_and_split_parameters_are_used():
    rng = np.random.default_rng(21)
    xs = rng.random(60)
    alpha, grid, c, theta = 0.1, 25, 0.75, 0.3
    seq = betting_cs(xs, alpha, grid=grid, c=c, theta=theta, intersect=False)
    accepted = [
        j / grid
        for j in range(grid + 1)
        if reference_capital(xs, j / grid, alpha, c=c, theta=theta) < 1.0 / alpha
    ]
    assert seq.lower[-1] == pytest.approx(max(min(accepted) - 1.0 / grid, 0.0), abs=1e-12)
    assert seq.upper[-1] == pytest.approx(min(max(accepted) + 1.0 / grid, 1.0), abs=1e-12)


def draw(name: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if name == "coin":
        return (rng.random(n) < 0.5).astype(float)
    if name == "rare":
        return (rng.random(n) < 0.1).astype(float)
    if name == "graded":
        return rng.beta(8.0, 2.0, size=n)
    raise AssertionError(name)


@pytest.mark.parametrize(("name", "mean"), [("coin", 0.5), ("rare", 0.1), ("graded", 0.8)])
def test_coverage_holds_at_every_look(name, mean):
    rng = np.random.default_rng(424242)
    alpha, reps, n = 0.1, 400, 250
    misses = 0
    for _ in range(reps):
        # grid=20 puts each true mean exactly on a grid point, so a miss is
        # exactly the event that the capital at the true mean reached 1 / alpha.
        seq = betting_cs(draw(name, rng, n), alpha, grid=20)
        misses += seq.threshold_stop(mean) is not None
    # Tolerance: three binomial standard errors above alpha.
    assert misses / reps <= alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / reps)


def test_running_intersection_is_nested_even_for_drifting_scores():
    rng = np.random.default_rng(9)
    scores = np.concatenate([rng.random(150) < 0.2, rng.random(150) < 0.7]).astype(float)
    seq = betting_cs(scores, 0.05, grid=200)
    assert np.all(np.diff(seq.lower) >= 0)
    assert np.all(np.diff(seq.upper) <= 0)
    raw = betting_cs(scores, 0.05, grid=200, intersect=False)
    assert np.any(np.diff(raw.upper) > 0)


def test_limits_bracket_the_mean_and_stay_in_bounds_for_stationary_scores():
    rng = np.random.default_rng(10)
    seq = betting_cs(rng.random(400) < 0.3, 0.05, grid=500)
    assert np.all((seq.lower >= 0.0) & (seq.upper <= 1.0))
    assert np.all(seq.lower <= seq.upper)
    assert seq.first_empty is None
    assert seq.lower[-1] <= seq.mean[-1] <= seq.upper[-1]
    assert seq.method == "betting"
    assert seq.alpha == 0.05


def test_narrower_than_the_mixture_when_the_variance_is_small():
    rng = np.random.default_rng(12)
    scores = (rng.random(2000) < 0.95).astype(float)
    betting = betting_cs(scores, 0.05)
    mixture = mixture_cs(scores, 0.05)
    assert betting.width[-1] < 0.75 * mixture.width[-1]


def test_a_run_of_passes_pushes_the_lower_limit_up():
    seq = betting_cs(np.ones(100), 0.05)
    assert seq.upper[-1] == 1.0
    assert 0.9 <= seq.lower[-1] < 1.0
    stop = seq.threshold_stop(0.5)
    assert stop is not None and stop.decision == "above"
    assert stop.n < 20


def test_other_score_ranges_are_an_affine_map_of_the_unit_case():
    rng = np.random.default_rng(13)
    unit = rng.random(120)
    base = betting_cs(unit, 0.05, grid=100)
    shifted = betting_cs(2.0 * unit - 1.0, 0.05, lo=-1.0, hi=1.0, grid=100)
    np.testing.assert_allclose(shifted.lower, 2.0 * base.lower - 1.0, atol=1e-12)
    np.testing.assert_allclose(shifted.upper, 2.0 * base.upper - 1.0, atol=1e-12)
    np.testing.assert_allclose(shifted.mean, 2.0 * base.mean - 1.0, atol=1e-12)


def test_online_updates_match_the_batch_function():
    rng = np.random.default_rng(14)
    scores = rng.random(90)
    batch = betting_cs(scores, 0.1, grid=64)
    online = BettingCS(0.1, grid=64)
    for t, score in enumerate(scores, start=1):
        assert online.update(score) == (batch.lower[t - 1], batch.upper[t - 1])
    assert online.n == 90


def test_a_grid_too_coarse_to_hold_the_set_still_brackets_the_mean():
    # Scores with mean 0.25 and a grid of 0, 0.5, 1: in the end all three grid
    # points are rejected and the set lies strictly between 0 and 0.5.
    scores = np.tile([0.0, 0.0, 0.0, 1.0], 150)
    alpha = 0.05
    assert all(reference_capital(scores, m, alpha) >= 1.0 / alpha for m in (0.0, 0.5, 1.0))
    seq = betting_cs(scores, alpha, grid=2)
    assert seq.lower[-1] <= 0.25 <= seq.upper[-1]
    assert seq.first_empty is None


def test_scores_that_change_level_empty_the_running_intersection():
    scores = np.concatenate([np.zeros(300), np.ones(300)])
    seq = betting_cs(scores, 0.05, grid=200)
    assert seq.first_empty is not None and seq.first_empty > 300
    stop = seq.threshold_stop(0.5)
    assert stop is not None and stop.decision == "below" and stop.n <= 300


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"lo": 2.0, "hi": 1.0}, "bounds"),
        ({"grid": 1}, "grid"),
        ({"grid": 10.5}, "grid"),
        ({"c": 1.0}, "c must"),
        ({"c": 0.0}, "c must"),
        ({"theta": 0.0}, "theta"),
        ({"theta": 1.0}, "theta"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BettingCS(**kwargs)


@pytest.mark.parametrize("bad", [1.01, -0.5, math.nan, math.inf, "high", None])
def test_invalid_scores_are_rejected(bad):
    with pytest.raises(ValueError):
        BettingCS().update(bad)
    with pytest.raises(ValueError):
        betting_cs([0.5, bad])
