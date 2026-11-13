"""Numerical helpers against scipy and closed forms."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special, stats

from evalstop.stats import (
    discordant_log_evalue,
    mcnemar_critical,
    mcnemar_exact_p,
    norm_cdf,
    norm_ppf,
)


def test_normal_functions_match_scipy():
    x = np.linspace(-8.0, 8.0, 161)
    np.testing.assert_allclose(norm_cdf(x), stats.norm.cdf(x), rtol=1e-12, atol=1e-300)
    for p in (1e-6, 0.025, 0.5, 0.9, 0.999):
        assert norm_ppf(p) == pytest.approx(stats.norm.ppf(p), abs=1e-9)
    assert norm_cdf(0.0) == pytest.approx(0.5)


@pytest.mark.parametrize(("wins", "losses"), [(0, 0), (6, 0), (0, 6), (7, 7), (58, 28), (3, 11)])
def test_exact_mcnemar_matches_the_binomial_test(wins, losses):
    k = wins + losses
    expected = 1.0 if k == 0 else stats.binomtest(wins, k, 0.5).pvalue
    assert mcnemar_exact_p(wins, losses) == pytest.approx(expected, rel=1e-9)


def test_exact_mcnemar_hand_worked_values():
    assert mcnemar_exact_p(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_exact_p(1, 4) == pytest.approx(2 * (1 + 5) / 32)
    with pytest.raises(ValueError, match="negative"):
        mcnemar_exact_p(-1, 2)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_critical_values_agree_with_the_p_value(alpha):
    crit = mcnemar_critical(alpha, 70)
    assert crit[0] == -1
    for k in range(71):
        for low in range(k // 2 + 1):
            assert (low <= crit[k]) == (mcnemar_exact_p(low, k - low) <= alpha)


def test_log_evalue_closed_form_for_the_uniform_prior():
    # 2**k * w! * l! / (k + 1)!
    for wins, losses in [(0, 0), (8, 0), (3, 5), (20, 7)]:
        k = wins + losses
        expected = 2.0**k * math.factorial(wins) * math.factorial(losses) / math.factorial(k + 1)
        assert math.exp(float(discordant_log_evalue(wins, losses))) == pytest.approx(expected)


@pytest.mark.parametrize("prior", [0.5, 1.0, 2.5])
def test_log_evalue_matches_the_beta_function_and_is_a_martingale(prior):
    wins = np.array([0, 1, 4, 12, 30])
    losses = np.array([0, 3, 4, 2, 41])
    expected = (
        (wins + losses) * math.log(2.0)
        + special.betaln(prior + wins, prior + losses)
        - special.betaln(prior, prior)
    )
    np.testing.assert_allclose(discordant_log_evalue(wins, losses, prior), expected, atol=1e-10)
    # Under the null the next pair is a win or a loss with probability 1/2 each.
    now = np.exp(discordant_log_evalue(wins, losses, prior))
    after_win = np.exp(discordant_log_evalue(wins + 1, losses, prior))
    after_loss = np.exp(discordant_log_evalue(wins, losses + 1, prior))
    np.testing.assert_allclose((after_win + after_loss) / 2.0, now, rtol=1e-10)


def test_log_evalue_rejects_a_bad_prior():
    with pytest.raises(ValueError, match="prior"):
        discordant_log_evalue(1, 1, 0.0)
