"""Vectorised stopping rules: agreement with the sequences, hand cases, error rates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalstop import betting_cs, mixture_cs, rules


def paired_paths(rng, runs, n, effect=0.0, discordance=0.3):
    u = rng.random((runs, n))
    p_plus = (discordance + effect) / 2.0
    return (u < p_plus).astype(float) - ((u >= p_plus) & (u < discordance)).astype(float)


def expected_from_sequence(seq, min_effect=None):
    """Stop and decision implied by a confidence sequence on the differences."""
    n = len(seq)
    settled = seq.threshold_stop(0.0)
    futile = None if min_effect is None else seq.inside_stop(-min_effect, min_effect)
    if settled is not None and (futile is None or settled.n <= futile.n):
        return settled.n, rules.A_BETTER if settled.decision == "above" else rules.B_BETTER
    if futile is not None:
        return futile.n, rules.FUTILE
    return n, rules.UNDECIDED


@pytest.mark.parametrize("min_effect", [None, 0.1])
def test_betting_rule_stops_exactly_when_the_sequence_does(min_effect):
    rng = np.random.default_rng(31)
    paths = np.vstack(
        [
            paired_paths(rng, 8, 160, effect=0.0),
            paired_paths(rng, 8, 160, effect=0.25),
            paired_paths(rng, 8, 160, effect=-0.25, discordance=0.6),
            paired_paths(rng, 6, 160, effect=0.0, discordance=0.05),
        ]
    )
    outcome = rules.betting_rule(paths, 0.05, min_effect=min_effect)
    seen = set()
    for row, path in enumerate(paths):
        seq = betting_cs(path, 0.05, lo=-1.0, hi=1.0, grid=200)
        stop, decision = expected_from_sequence(seq, min_effect)
        assert (int(outcome.stop[row]), int(outcome.decision[row])) == (stop, decision)
        seen.add(decision)
    assert {rules.A_BETTER, rules.B_BETTER, rules.UNDECIDED} <= seen
    if min_effect is not None:
        assert rules.FUTILE in seen


def test_betting_rule_handles_graded_scores_and_other_ranges():
    rng = np.random.default_rng(32)
    paths = rng.uniform(-4.0, 4.0, size=(10, 120)) + 0.8
    paths = np.clip(paths, -4.0, 4.0)
    outcome = rules.betting_rule(paths, 0.1, scale=4.0, min_effect=2.0)
    for row, path in enumerate(paths):
        seq = betting_cs(path, 0.1, lo=-4.0, hi=4.0, grid=400)
        assert (int(outcome.stop[row]), int(outcome.decision[row])) == expected_from_sequence(
            seq, 2.0
        )


@pytest.mark.parametrize("min_effect", [None, 0.3])
def test_mixture_rule_stops_exactly_when_the_sequence_does(min_effect):
    rng = np.random.default_rng(33)
    paths = np.vstack(
        [
            paired_paths(rng, 10, 400, effect=0.0, discordance=0.2),
            paired_paths(rng, 10, 400, effect=0.5, discordance=0.7),
            paired_paths(rng, 10, 400, effect=-0.5, discordance=0.7),
        ]
    )
    outcome = rules.mixture_rule(paths, 0.05, tune_at=100, min_effect=min_effect)
    seen = set()
    for row, path in enumerate(paths):
        seq = mixture_cs(path, 0.05, lo=-1.0, hi=1.0, tune_at=100)
        stop, decision = expected_from_sequence(seq, min_effect)
        assert (int(outcome.stop[row]), int(outcome.decision[row])) == (stop, decision)
        seen.add(decision)
    assert {rules.A_BETTER, rules.B_BETTER} <= seen
    if min_effect is not None:
        assert rules.FUTILE in seen


def test_discordant_rule_hand_worked_stop():
    # With only wins the evidence ratio is 2**k / (k + 1): 16 after 7, 28.4 after 8.
    path = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    outcome = rules.discordant_rule(path, 0.05)
    assert (int(outcome.stop[0]), int(outcome.decision[0])) == (10, rules.A_BETTER)
    outcome = rules.discordant_rule(-path, 0.05)
    assert (int(outcome.stop[0]), int(outcome.decision[0])) == (10, rules.B_BETTER)
    # At alpha = 0.1 the threshold is 10: 2**6 / 7 = 9.14 is short of it, 2**7 / 8 = 16 is not.
    outcome = rules.discordant_rule(path, 0.1)
    assert int(outcome.stop[0]) == 9
    balanced = np.tile([1.0, -1.0], 100)
    outcome = rules.discordant_rule(balanced, 0.05)
    assert (int(outcome.stop[0]), int(outcome.decision[0])) == (200, rules.UNDECIDED)


def test_naive_and_fixed_rules_hand_worked_stops():
    # Six wins in a row give the exact p-value 2 / 64 = 0.03125.
    path = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0], dtype=float)
    naive = rules.naive_rule(path, 0.05)
    assert (int(naive.stop[0]), int(naive.decision[0])) == (7, rules.A_BETTER)
    spaced = rules.naive_rule(path, 0.05, every=4)
    assert (int(spaced.stop[0]), int(spaced.decision[0])) == (8, rules.A_BETTER)
    fixed = rules.fixed_rule(path, 0.05)
    assert (int(fixed.stop[0]), int(fixed.decision[0])) == (10, rules.A_BETTER)
    # A lead that fades: significant at an early look, not at the end.
    fading = np.array([-1.0] * 6 + [1.0] * 6)
    assert int(rules.naive_rule(fading, 0.05).decision[0]) == rules.B_BETTER
    assert int(rules.fixed_rule(fading, 0.05).decision[0]) == rules.UNDECIDED
    assert int(rules.fixed_rule(fading, 0.05).stop[0]) == 12


def test_error_rates_under_the_null():
    rng = np.random.default_rng(34)
    alpha, runs, n = 0.05, 2500, 300
    paths = paired_paths(rng, runs, n, effect=0.0)
    limit = alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / runs)
    rates = {}
    for name in rules.RULES:
        outcome = rules.run_rule(name, paths, alpha)
        assert outcome.stop.min() >= 1 and outcome.stop.max() <= n
        assert np.all(outcome.stop[outcome.decision == rules.UNDECIDED] == n)
        picked = (outcome.decision == rules.A_BETTER) | (outcome.decision == rules.B_BETTER)
        rates[name] = float(np.mean(picked))
    # Tolerance: three binomial standard errors above alpha.
    for name in ("betting", "mixture", "discordant", "fixed"):
        assert rates[name] <= limit, (name, rates[name])
    # The group-sequential boundaries rest on a normal approximation: allow half a point more.
    for name in ("obf", "pocock"):
        assert rates[name] <= limit + 0.005, (name, rates[name])
    # Wald's thresholds bound the error rate by alpha / (1 - beta), here with beta = 0.2.
    assert rates["sprt"] <= alpha / 0.8 + 3.0 * math.sqrt(alpha * (1.0 - alpha) / runs)
    # Peeking after every item is far above its nominal level.
    assert rates["naive"] > 3.0 * alpha


def test_power_and_futility():
    rng = np.random.default_rng(35)
    runs, n = 400, 600
    effect = paired_paths(rng, runs, n, effect=0.15)
    for name in ("betting", "discordant"):
        outcome = rules.run_rule(name, effect, 0.05)
        assert np.mean(outcome.decision == rules.A_BETTER) > 0.9
        assert np.mean(outcome.decision == rules.B_BETTER) == 0.0
        assert outcome.stop.mean() < 0.6 * n
    null = paired_paths(rng, runs, 1500, effect=0.0, discordance=0.2)
    futile = rules.betting_rule(null, 0.05, min_effect=0.15)
    assert np.mean(futile.decision == rules.FUTILE) > 0.8
    assert futile.stop.mean() < 1500
    # A true difference of twice the margin must almost never be called futile.
    strong = paired_paths(rng, runs, 600, effect=0.3, discordance=0.4)
    outcome = rules.betting_rule(strong, 0.05, min_effect=0.15)
    assert np.mean(outcome.decision == rules.FUTILE) <= 0.05


def test_look_items():
    assert rules.look_items(500, 5) == [100, 200, 300, 400, 500]
    assert rules.look_items(10, 4) == [3, 5, 8, 10]
    assert rules.look_items(3, 5) == [1, 2, 3]
    with pytest.raises(ValueError, match="looks"):
        rules.look_items(10, 0)


def test_group_sequential_rule_hand_worked_stops():
    # 100 items, looks after 20, 40, ... With only wins z = sqrt(k): 4.47 after 20 items is
    # short of the first O'Brien-Fleming boundary 4.88 but beyond the Pocock boundary 2.44.
    wins = np.ones(100)
    obf = rules.group_sequential_rule(wins, 0.05, looks=5, spending="obf")
    assert (int(obf.stop[0]), int(obf.decision[0])) == (40, rules.A_BETTER)
    pocock = rules.group_sequential_rule(-wins, 0.05, looks=5, spending="pocock")
    assert (int(pocock.stop[0]), int(pocock.decision[0])) == (20, rules.B_BETTER)
    # A lead inside one group is not seen until the next planned look.
    late = np.concatenate([np.zeros(21), np.ones(79)])
    outcome = rules.group_sequential_rule(late, 0.05, looks=5, spending="pocock")
    assert int(outcome.stop[0]) == 40
    ties = rules.group_sequential_rule(np.zeros(50), 0.05)
    assert (int(ties.stop[0]), int(ties.decision[0])) == (50, rules.UNDECIDED)


def test_sprt_rule_hand_worked_stops():
    # Thresholds at alpha / 2 = 0.025 and beta = 0.2: log(0.8 / 0.025) = 3.466 at the top,
    # log(0.2 / 0.975) = -1.584 at the bottom. A win adds log(1.3) = 0.262 to the test for A.
    needed = math.ceil(math.log(0.8 / 0.025) / math.log(1.3))
    assert needed == 14
    outcome = rules.sprt_rule(np.ones(40), 0.05, beta=0.2, win_rate=0.65)
    assert (int(outcome.stop[0]), int(outcome.decision[0])) == (14, rules.A_BETTER)
    outcome = rules.sprt_rule(-np.ones(40), 0.05, beta=0.2, win_rate=0.65)
    assert (int(outcome.stop[0]), int(outcome.decision[0])) == (14, rules.B_BETTER)
    # Alternating wins and losses: each pair adds log(1.3) + log(0.7) = -0.0943 to both
    # tests. The test for A reaches -1.584 after 17 pairs (item 34); the test for B dips
    # further after each win and gets there at item 29. Futility needs both.
    alternating = np.tile([1.0, -1.0], 30)
    outcome = rules.sprt_rule(alternating, 0.05, beta=0.2, win_rate=0.65)
    assert (int(outcome.stop[0]), int(outcome.decision[0])) == (34, rules.FUTILE)
    with pytest.raises(ValueError, match="win_rate"):
        rules.sprt_rule(alternating, 0.05, win_rate=0.5)


def test_summarise():
    outcome = rules.Outcome(
        np.array([10, 20, 30, 40]),
        np.array([rules.A_BETTER, rules.B_BETTER, rules.FUTILE, rules.UNDECIDED], dtype=np.int8),
    )
    summary = rules.summarise(outcome, 40)
    assert summary == {
        "a_better": 0.25,
        "b_better": 0.25,
        "futile": 0.25,
        "undecided": 0.25,
        "mean_items": 25.0,
        "median_items": 25.0,
        "mean_share": 0.625,
    }


def test_invalid_arguments_are_rejected():
    path = np.zeros(5)
    with pytest.raises(ValueError, match="unknown rule"):
        rules.run_rule("exact", path)
    with pytest.raises(ValueError, match="alpha"):
        rules.betting_rule(path, 0.0)
    with pytest.raises(ValueError, match="min_effect"):
        rules.betting_rule(path, 0.05, min_effect=1.0)
    with pytest.raises(ValueError, match="min_effect"):
        rules.mixture_rule(path, 0.05, min_effect=0.0)
    with pytest.raises(ValueError, match="between -scale and scale"):
        rules.betting_rule(np.array([0.5, 2.0]), 0.05)
    with pytest.raises(ValueError, match="every"):
        rules.naive_rule(path, 0.05, every=0)
    with pytest.raises(ValueError, match="at least one item"):
        rules.fixed_rule(np.zeros((3, 0)))
    with pytest.raises(ValueError, match="finite"):
        rules.discordant_rule(np.array([0.0, np.nan]))
