"""Betting sequence for a fixed benchmark seen in random order (without replacement)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from evalstop import BettingCS, betting_cs
from evalstop.cli import main
from evalstop.monitor import monitor_scores
from evalstop.paired import Pairs, compare_pairs


def reference_capital(xs, m: float, alpha: float, population: int) -> float:
    """Without-replacement hedged capital at candidate mean ``m``, from the formulas.

    Returns infinity for a candidate the scores seen so far make impossible.
    """
    c, theta = 0.5, 0.5
    k_plus = k_minus = 1.0
    total = 0.0
    sq_dev = 0.0
    for t, x in enumerate(xs, start=1):
        variance = (0.25 + sq_dev) / t
        bet = math.sqrt(2.0 * math.log(2.0 / alpha) / (variance * t * math.log(t + 1)))
        mean_next = (population * m - total) / (population - t + 1)
        if not -1e-9 <= mean_next <= 1 + 1e-9:
            return math.inf
        mean_next = min(max(mean_next, 0.0), 1.0)
        lam_plus = min(bet, c / mean_next) if mean_next > 0 else bet
        lam_minus = min(bet, c / (1 - mean_next)) if mean_next < 1 else bet
        k_plus *= 1 + lam_plus * (x - mean_next)
        k_minus *= 1 - lam_minus * (x - mean_next)
        total += x
        mu_hat = (0.5 + total) / (t + 1)
        sq_dev += (x - mu_hat) ** 2
    remaining = population - len(xs)
    if not total - 1e-9 <= population * m <= total + remaining + 1e-9:
        return math.inf
    return max(theta * k_plus, (1 - theta) * k_minus)


def test_interval_is_the_accepted_grid_points_widened_by_one_cell():
    rng = np.random.default_rng(51)
    population, grid, alpha = 60, 30, 0.2
    xs = rng.permutation(np.array([1.0] * 36 + [0.0] * 24))
    cs = BettingCS(alpha, grid=grid, intersect=False, population=population)
    for t, x in enumerate(xs, start=1):
        lower, upper = cs.update(x)
        accepted = [
            j / grid
            for j in range(grid + 1)
            if reference_capital(xs[:t], j / grid, alpha, population) < 1.0 / alpha
        ]
        assert accepted
        assert lower == pytest.approx(max(min(accepted) - 1.0 / grid, 0.0), abs=1e-12)
        assert upper == pytest.approx(min(max(accepted) + 1.0 / grid, 1.0), abs=1e-12)
    assert accepted == [pytest.approx(0.6)]


def test_coverage_of_the_benchmark_mean_over_random_orders():
    rng = np.random.default_rng(52)
    alpha, orders = 0.1, 80
    values = np.array([1.0] * 60 + [0.0] * 40)
    misses = 0
    for _ in range(orders):
        seq = betting_cs(rng.permutation(values), alpha, grid=20, population=100)
        misses += seq.threshold_stop(0.6) is not None
    # Tolerance: three binomial standard errors above alpha.
    assert misses / orders <= alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / orders)


@pytest.mark.parametrize(("ones", "zeros", "grid"), [(120, 80, 200), (7, 6, 10), (3, 0, 4)])
def test_interval_closes_on_the_exact_mean_at_the_end(ones, zeros, grid):
    rng = np.random.default_rng(53)
    values = rng.permutation(np.array([1.0] * ones + [0.0] * zeros))
    seq = betting_cs(values, 0.05, grid=grid, population=ones + zeros)
    mean = ones / (ones + zeros)
    assert seq.lower[-1] <= mean <= seq.upper[-1]
    assert seq.upper[-1] - seq.lower[-1] <= 2.0 / grid + 1e-12
    assert np.all(np.diff(seq.lower) >= 0) and np.all(np.diff(seq.upper) <= 0)


def test_narrower_than_the_with_replacement_sequence_late_in_the_run():
    rng = np.random.default_rng(54)
    values = rng.permutation(np.array([1.0] * 300 + [0.0] * 200))
    finite = betting_cs(values, 0.05, population=500)
    plain = betting_cs(values, 0.05)
    assert finite.width[449] < 0.6 * plain.width[449]
    assert finite.width[-1] <= 0.002 + 1e-12 < plain.width[-1]


def test_graded_scores_and_other_ranges():
    rng = np.random.default_rng(55)
    values = np.round(rng.uniform(2.0, 9.0, 40), 1)
    mean = float(values.mean())
    for _ in range(40):
        seq = betting_cs(rng.permutation(values), 0.1, lo=0.0, hi=10.0, grid=100, population=40)
        assert seq.lower[-1] <= mean <= seq.upper[-1]
        assert seq.upper[-1] - seq.lower[-1] <= 0.2 + 1e-9


def test_population_errors():
    with pytest.raises(ValueError, match="population"):
        BettingCS(population=0)
    with pytest.raises(ValueError, match="population"):
        BettingCS(population=2.5)
    cs = BettingCS(population=2)
    cs.update(1.0)
    cs.update(0.0)
    with pytest.raises(ValueError, match="more scores than the population of 2 items"):
        cs.update(1.0)
    with pytest.raises(ValueError, match="only supported by the betting method"):
        monitor_scores([0, 1], method="mixture", population=2)
    pairs = Pairs("a", "b", ("x", "y"), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="only supported by the betting method"):
        compare_pairs(pairs, method="mixture", population=2)


def test_population_option_in_the_commands(tmp_path, capsys):
    rng = np.random.default_rng(56)
    a = rng.permutation(np.array([1] * 150 + [0] * 50))
    b = rng.permutation(np.array([1] * 100 + [0] * 100))
    rows = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        rows.append({"item": f"q{i}", "model": "a", "score": int(x)})
        rows.append({"item": f"q{i}", "model": "b", "score": int(y)})
    path = tmp_path / "bench.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert main(["monitor", str(path), "--model", "a", "--population", "200",
                 "--format", "json"]) == 0  # fmt: skip
    data = json.loads(capsys.readouterr().out)
    assert data["settings"] == {"grid": 1000, "population": 200}
    assert data["lower"] == pytest.approx(0.749) and data["upper"] == pytest.approx(0.751)

    assert main(["compare", str(path), "--population", "200"]) == 0
    out = capsys.readouterr().out
    assert "betting sequence on the difference over the whole benchmark" in out
    assert "population 200" in out
    diff = float(np.mean(a - b))
    assert f"[{diff - 0.002:+.4f}, {diff + 0.002:+.4f}]" in out

    assert main(["monitor", str(path), "--model", "a", "--population", "100"]) == 2
    assert "more scores than the population of 100 items" in capsys.readouterr().err
