"""Counts in the report text use the singular for one."""

from __future__ import annotations

import numpy as np

from evalstop.monitor import monitor_scores
from evalstop.paired import Pairs, compare_pairs
from evalstop.simulate import replay, simulate
from evalstop.text import count


def test_count():
    assert count(1, "score") == "1 score"
    assert count(0, "score") == "0 scores"
    assert count(12, "pair") == "12 pairs"


def test_monitor_report_with_one_score():
    text = monitor_scores([1.0], threshold=0.5).to_markdown()
    assert "confidence sequence after 1 score: " in text
    assert "| threshold 0.5 | not within 1 score | undecided | |" in text
    assert "1 scores" not in text
    many = monitor_scores([1.0] * 30, threshold=0.5).to_markdown()
    assert "confidence sequence after 30 scores: " in many and " of 30 scores | " in many


def test_monitor_warning_counts_scores():
    text = monitor_scores([0.0] * 300 + [1.0] * 300, grid=200).to_markdown()
    assert " scores the running intersection became empty. The scores do not look" in text


def test_compare_report_with_one_pair():
    pairs = Pairs("a", "b", ("q1",), np.array([1.0]), np.array([0.0]), unpaired=1)
    text = compare_pairs(pairs).to_markdown()
    assert "for the difference after 1 pair: " in text
    assert "| not within 1 pair | undecided" in text
    assert "- left out: 1 score for items only one model was run on" in text
    assert "1 pairs" not in text and "1 scores" not in text
    two = Pairs("a", "b", ("q1", "q2"), np.array([1.0, 0.0]), np.array([0.0, 0.0]), unpaired=2)
    text = compare_pairs(two).to_markdown()
    assert "after 2 pairs: " in text and "- left out: 2 scores for items" in text


def test_simulate_and_replay_headers_with_one_run():
    text = simulate(effects=(0.0,), items=1, runs=1, rule_names=("discordant",)).to_markdown()
    assert "- 1 run of 1 paired pass/fail item per effect, seed 0" in text
    text = simulate(effects=(0.0,), items=2, runs=3, rule_names=("discordant",)).to_markdown()
    assert "- 3 runs of 2 paired pass/fail items per effect, seed 0" in text
    pairs = Pairs("a", "b", ("q1", "q2"), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    assert "- 1 random order, seed 0" in replay(pairs, orders=1).to_markdown()
    assert "- 2 random orders, seed 0" in replay(pairs, orders=2).to_markdown()
