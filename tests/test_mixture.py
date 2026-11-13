"""Normal-mixture confidence sequence: closed forms, an independent reference, simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import lambertw

from evalstop import MixtureCS, best_rho, mixture_cs, normal_mixture_boundary


def mixture_value(s: float, v: float, rho: float) -> float:
    """Closed form of the N(0, 1/rho) mixture of exp(lam * s - lam^2 * v / 2)."""
    return math.sqrt(rho / (v + rho)) * math.exp(s * s / (2.0 * (v + rho)))


def test_boundary_hand_worked_value():
    # rho = 1, v = 3, alpha = 0.5: sqrt(4 * log(4 / 0.25)) = sqrt(4 * log(16))
    expected = math.sqrt(4 * math.log(16))
    assert float(normal_mixture_boundary(3.0, 0.5, 1.0)) == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("v", [0.25, 10.0, 400.0])
@pytest.mark.parametrize("rho", [0.5, 7.0])
def test_boundary_is_where_the_mixture_reaches_one_over_alpha(alpha, v, rho):
    s = float(normal_mixture_boundary(v, alpha, rho))
    assert mixture_value(s, v, rho) == pytest.approx(1.0 / alpha, rel=1e-10)


def test_mixture_closed_form_matches_numerical_integration():
    s, v, rho = 3.0, 12.0, 2.5

    def integrand(lam: float) -> float:
        density = math.sqrt(rho / (2.0 * math.pi)) * math.exp(-rho * lam * lam / 2.0)
        return math.exp(lam * s - lam * lam * v / 2.0) * density

    numeric, _ = quad(integrand, -np.inf, np.inf)
    assert numeric == pytest.approx(mixture_value(s, v, rho), rel=1e-8)


@pytest.mark.parametrize("alpha", [0.001, 0.05, 0.3])
def test_best_rho_matches_lambert_w(alpha):
    v_opt = 37.0
    expected = v_opt / (-lambertw(-(alpha**2) / math.e, k=-1).real - 1.0)
    assert best_rho(v_opt, alpha) == pytest.approx(expected, rel=1e-10)


def test_best_rho_minimises_the_boundary_at_the_chosen_time():
    alpha, v_opt = 0.05, 50.0
    rho = best_rho(v_opt, alpha)
    best = float(normal_mixture_boundary(v_opt, alpha, rho))
    for factor in (0.5, 0.9, 1.1, 2.0):
        assert float(normal_mixture_boundary(v_opt, alpha, rho * factor)) > best


def test_radius_is_close_to_hoeffding_at_the_tuned_time_and_shrinks():
    alpha, n = 0.05, 200
    seq = mixture_cs(np.full(n, 0.5), alpha=alpha, tune_at=n, intersect=False)
    radius = (seq.upper[-1] - seq.lower[-1]) / 2.0
    hoeffding = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    assert hoeffding < radius < 1.25 * hoeffding
    long = mixture_cs(np.full(200_000, 0.5), alpha=alpha, tune_at=n)
    assert long.width[-1] < 0.01


@pytest.mark.parametrize("p", [0.05, 0.5, 0.9])
def test_coverage_holds_at_every_look(p):
    rng = np.random.default_rng(20260101)
    alpha, reps, n = 0.1, 2000, 1000
    misses = 0
    for _ in range(reps):
        seq = mixture_cs(rng.random(n) < p, alpha=alpha, tune_at=100, intersect=False)
        misses += bool(np.any((seq.lower >= p) | (seq.upper <= p)))
    # Tolerance: three binomial standard errors above alpha.
    assert misses / reps <= alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / reps)


def test_fair_coin_uses_a_real_share_of_alpha():
    # The sub-Gaussian bound is nearly tight for a fair coin, so the sequence
    # should not be far more conservative than its level.
    rng = np.random.default_rng(5)
    alpha, reps, n = 0.1, 2000, 2000
    misses = 0
    for _ in range(reps):
        seq = mixture_cs(rng.random(n) < 0.5, alpha=alpha, tune_at=50)
        misses += seq.threshold_stop(0.5) is not None
    assert misses / reps >= alpha / 4.0


def test_running_intersection_is_nested():
    rng = np.random.default_rng(3)
    seq = mixture_cs(rng.random(500), alpha=0.05)
    assert np.all(np.diff(seq.lower) >= 0)
    assert np.all(np.diff(seq.upper) <= 0)
    raw = mixture_cs(rng.random(500), alpha=0.05, intersect=False)
    assert np.any(np.diff(raw.lower) < 0)


def test_online_updates_match_the_batch_function():
    rng = np.random.default_rng(11)
    scores = rng.random(300)
    batch = mixture_cs(scores, alpha=0.1, tune_at=80)
    online = MixtureCS(0.1, tune_at=80)
    for t, score in enumerate(scores, start=1):
        lower, upper = online.update(score)
        assert lower == pytest.approx(batch.lower[t - 1], abs=1e-12)
        assert upper == pytest.approx(batch.upper[t - 1], abs=1e-12)
        assert online.mean == pytest.approx(batch.mean[t - 1], abs=1e-12)
    assert online.n == 300
    again = MixtureCS(0.1, tune_at=80).run(scores)
    np.testing.assert_allclose(again.lower, batch.lower, atol=1e-12)
    np.testing.assert_allclose(again.upper, batch.upper, atol=1e-12)
    assert again.method == batch.method == "mixture"


def test_other_score_ranges_are_an_affine_map_of_the_unit_case():
    rng = np.random.default_rng(2)
    unit = rng.random(200)
    base = mixture_cs(unit, alpha=0.05)
    shifted = mixture_cs(2.0 * unit - 1.0, alpha=0.05, lo=-1.0, hi=1.0)
    np.testing.assert_allclose(shifted.lower, 2.0 * base.lower - 1.0, atol=1e-12)
    np.testing.assert_allclose(shifted.upper, 2.0 * base.upper - 1.0, atol=1e-12)
    np.testing.assert_allclose(shifted.mean, 2.0 * base.mean - 1.0, atol=1e-12)
    assert shifted.bounds == (-1.0, 1.0)


def test_limits_stay_inside_the_score_bounds():
    seq = mixture_cs([1, 1, 1, 0, 1], alpha=0.05)
    assert np.all(seq.lower >= 0.0)
    assert np.all(seq.upper <= 1.0)
    assert len(seq) == 5


def test_empty_input_gives_an_empty_sequence():
    seq = mixture_cs([], alpha=0.05)
    assert len(seq) == 0
    assert seq.threshold_stop(0.5) is None
    assert seq.width_stop(0.1) is None


def test_generators_are_accepted():
    seq = mixture_cs((x for x in [0.0, 1.0, 1.0]), alpha=0.05)
    assert len(seq) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"lo": 1.0, "hi": 1.0}, "bounds"),
        ({"lo": 0.0, "hi": math.inf}, "bounds"),
        ({"tune_at": 0}, "tune_at"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mixture_cs([0.5], **kwargs)
    with pytest.raises(ValueError, match=message):
        MixtureCS(**kwargs)


@pytest.mark.parametrize("bad", [[0.5, 1.5], [-0.1], [math.nan], [math.inf], [[0.1, 0.2]]])
def test_invalid_scores_are_rejected(bad):
    with pytest.raises(ValueError):
        mixture_cs(bad)


@pytest.mark.parametrize("bad", [1.5, -0.1, math.nan, math.inf])
def test_online_rejects_invalid_scores(bad):
    with pytest.raises(ValueError):
        MixtureCS().update(bad)


def test_boundary_and_rho_reject_invalid_arguments():
    with pytest.raises(ValueError, match="rho"):
        normal_mixture_boundary(1.0, 0.05, 0.0)
    with pytest.raises(ValueError, match="v_opt"):
        best_rho(0.0, 0.05)
    with pytest.raises(ValueError, match="alpha"):
        best_rho(1.0, 1.5)
