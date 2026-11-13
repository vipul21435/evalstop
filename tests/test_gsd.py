"""Alpha-spending boundaries: closed forms, an independent integral, simulation."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import stats

from evalstop.cli import main
from evalstop.gsd import DesignReport, design, spending


@pytest.mark.parametrize("kind", ["obf", "pocock"])
def test_spending_functions_end_at_alpha_and_increase(kind):
    values = [spending(kind, t, 0.05) for t in (0.1, 0.25, 0.5, 0.75, 1.0)]
    assert values[-1] == pytest.approx(0.05, abs=1e-12)
    assert all(a < b for a, b in zip(values, values[1:], strict=False))


def test_spending_closed_forms():
    assert spending("pocock", 0.5, 0.05) == pytest.approx(0.05 * math.log(1 + (math.e - 1) * 0.5))
    z = stats.norm.ppf(1 - 0.05 / 4)
    assert spending("obf", 0.5, 0.05) == pytest.approx(4 - 4 * stats.norm.cdf(z / math.sqrt(0.5)))
    with pytest.raises(ValueError, match="unknown spending function"):
        spending("linear", 0.5, 0.05)


def test_a_single_look_is_the_fixed_sample_test():
    plan = design(1, 0.05, "obf")
    assert plan.z == pytest.approx((stats.norm.ppf(0.975),), abs=1e-9)
    assert plan.nominal_p == pytest.approx((0.05,), abs=1e-9)


@pytest.mark.parametrize("kind", ["obf", "pocock"])
def test_first_boundary_spends_exactly_the_first_increment(kind):
    plan = design(4, 0.05, kind)
    assert plan.z[0] == pytest.approx(stats.norm.ppf(1 - spending(kind, 0.25, 0.05) / 2), abs=1e-9)
    assert plan.spent[-1] == pytest.approx(0.05)


@pytest.mark.parametrize("kind", ["obf", "pocock"])
@pytest.mark.parametrize("looks", [2, 3, 4, (0.3, 0.55, 1.0)])
def test_cumulative_crossing_probability_matches_scipy(kind, looks):
    """Independent check: integrate the joint normal distribution with scipy."""
    plan = design(looks, 0.05, kind)
    t = np.asarray(plan.fractions)
    z = np.asarray(plan.z)
    for k in range(2, len(t) + 1):
        cov = np.sqrt(np.minimum.outer(t[:k], t[:k]) / np.maximum.outer(t[:k], t[:k]))
        inside = stats.multivariate_normal(mean=np.zeros(k), cov=cov).cdf(
            z[:k], lower_limit=-z[:k]
        )
        assert 1.0 - inside == pytest.approx(plan.spent[k - 1], abs=2e-4)


def test_five_look_boundaries():
    # Values this implementation gives; the test above checks the method independently.
    obf = design(5, 0.05, "obf")
    assert obf.z == pytest.approx((4.8769, 3.3570, 2.6803, 2.2898, 2.0310), abs=5e-4)
    pocock = design(5, 0.05, "pocock")
    assert pocock.z == pytest.approx((2.4380, 2.4268, 2.4102, 2.3966, 2.3860), abs=5e-4)
    assert all(a > b for a, b in zip(obf.z, obf.z[1:], strict=False))
    assert max(pocock.z) - min(pocock.z) < 0.06


@pytest.mark.parametrize("kind", ["obf", "pocock"])
def test_simulated_brownian_paths_cross_with_probability_alpha(kind):
    rng = np.random.default_rng(2024)
    looks, alpha, runs = 5, 0.05, 200_000
    plan = design(looks, alpha, kind)
    steps = rng.standard_normal((runs, looks)) * math.sqrt(1.0 / looks)
    z = np.cumsum(steps, axis=1) / np.sqrt(np.asarray(plan.fractions))
    crossed = np.any(np.abs(z) >= np.asarray(plan.z), axis=1).mean()
    # Tolerance: three binomial standard errors either side of alpha.
    assert abs(crossed - alpha) <= 3.0 * math.sqrt(alpha * (1 - alpha) / runs)


@pytest.mark.parametrize("bad", [0, (0.5, 0.4, 1.0), (0.0, 1.0), (0.5, 1.2), ()])
def test_invalid_looks_are_rejected(bad):
    with pytest.raises(ValueError):
        design(bad)


def test_design_report_and_command(capsys):
    report = DesignReport(design(5, 0.05, "obf"), items=500)
    data = report.to_dict()
    assert [look["items"] for look in data["looks"]] == [100, 200, 300, 400, 500]
    assert data["looks"][-1]["alpha_spent"] == pytest.approx(0.05)
    text = report.to_markdown()
    assert "O'Brien-Fleming type spending" in text and "| 5 | 1.000 | 500 | 2.0310 |" in text
    assert "| 1 | 0.333 | - |" in DesignReport(design(3, 0.05, "pocock")).to_markdown()

    assert main(["design", "--looks", "3", "--spending", "pocock", "--format", "json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["spending"] == "pocock" and len(out["looks"]) == 3 and out["items"] is None
    assert main(["design", "--looks", "0"]) == 2
    assert "looks must be at least 1" in capsys.readouterr().err
