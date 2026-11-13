"""Wald's sequential probability ratio test against his closed forms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalstop.sprt import BernoulliSPRT, expected_sample_size, sprt_bernoulli, wald_thresholds


def test_thresholds_are_walds():
    upper, lower = wald_thresholds(0.05, 0.2)
    assert upper == pytest.approx(math.log(0.8 / 0.05))
    assert lower == pytest.approx(math.log(0.2 / 0.95))


@pytest.mark.parametrize(("p0", "p1"), [(0.5, 0.65), (0.3, 0.5), (0.6, 0.4)])
def test_decisions_follow_the_closed_form_lines(p0, p1):
    rng = np.random.default_rng(77)
    seen = set()
    for truth in (p0, p1):
        for _ in range(60):
            test = BernoulliSPRT(p0, p1, 0.05, 0.1)
            slope, h1, h0 = test.lines()
            passes = 0
            for n in range(1, 2001):
                passed = bool(rng.random() < truth)
                passes += passed
                decision = test.update(passed)
                above = passes >= h1 + slope * n - 1e-9
                below = passes <= h0 + slope * n + 1e-9
                if p1 < p0:
                    above = passes <= h1 + slope * n + 1e-9
                    below = passes >= h0 + slope * n - 1e-9
                expected = "h1" if above else "h0" if below else None
                assert decision == expected
                if decision is not None:
                    break
            seen.add(test.decision)
    assert seen == {"h0", "h1"}


def test_line_constants_hand_worked():
    # p0 = 1/2 against p1 = 2/3: log(p1/p0) = log(4/3), log((1-p1)/(1-p0)) = log(2/3).
    slope, h1, h0 = BernoulliSPRT(0.5, 2 / 3, 0.05, 0.2).lines()
    assert slope == pytest.approx(math.log(1.5) / math.log(2.0))
    assert h1 == pytest.approx(math.log(16.0) / math.log(2.0))
    assert h0 == pytest.approx(math.log(0.2 / 0.95) / math.log(2.0))


def test_error_rates_and_sample_size_against_walds_formulas():
    rng = np.random.default_rng(78)
    p0, p1, alpha, beta, runs = 0.5, 0.65, 0.05, 0.2, 3000
    results = {}
    for truth in (p0, p1):
        chose_h1 = 0
        used = 0
        for _ in range(runs):
            outcome = sprt_bernoulli(rng.random(1500) < truth, p0, p1, alpha, beta)
            assert outcome.decision is not None
            chose_h1 += outcome.decision == "h1"
            used += outcome.n
        results[truth] = (chose_h1 / runs, used / runs)
    type_one, mean_null = results[p0]
    power, mean_alt = results[p1]
    # Wald's bounds on the true error rates, plus three binomial standard errors.
    assert type_one <= alpha / (1 - beta) + 3 * math.sqrt(alpha * (1 - alpha) / runs)
    assert 1 - power <= beta / (1 - alpha) + 3 * math.sqrt(beta * (1 - beta) / runs)
    # Wald's approximation ignores the overshoot, so the true mean is somewhat larger.
    for mean, truth in ((mean_null, p0), (mean_alt, p1)):
        approx = expected_sample_size(truth, p0, p1, alpha, beta)
        assert 0.95 * approx <= mean <= 1.3 * approx


def test_expected_sample_size_hand_worked():
    upper, lower = math.log(16.0), math.log(0.2 / 0.95)
    drift = 0.5 * math.log(1.3) + 0.5 * math.log(0.7)
    expected = (0.95 * lower + 0.05 * upper) / drift
    assert expected_sample_size(0.5, 0.5, 0.65) == pytest.approx(expected)
    with pytest.raises(ValueError, match="p must equal"):
        expected_sample_size(0.55, 0.5, 0.65)


def test_the_test_stops_once_and_reports_its_path():
    outcome = sprt_bernoulli([1] * 50, 0.5, 0.65, 0.05, 0.2)
    needed = math.ceil(math.log(16.0) / math.log(1.3))
    assert (outcome.n, outcome.decision) == (needed, "h1")
    assert len(outcome.llr) == needed and outcome.llr[-1] >= math.log(16.0)
    short = sprt_bernoulli([1, 0, 1], 0.5, 0.65)
    assert short.decision is None and short.n == 3
    test = BernoulliSPRT(0.5, 0.65)
    for _ in range(needed):
        test.update(True)
    assert test.update(False) == "h1" and test.n == needed


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.5, 0.5), "must differ"),
        ((0.0, 0.5), "must differ and lie"),
        ((0.5, 0.65, 0.0), "alpha"),
        ((0.5, 0.65, 0.05, 1.0), "beta"),
        ((0.5, 0.65, 0.6, 0.5), "below 1"),
    ],
)
def test_invalid_arguments_are_rejected(args, message):
    with pytest.raises(ValueError, match=message):
        BernoulliSPRT(*args)
