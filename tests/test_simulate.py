"""simulate and replay."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from evalstop import rules
from evalstop.cli import main
from evalstop.paired import Pairs
from evalstop.simulate import replay, simulate
from evalstop.stats import mcnemar_exact_p


def make_pairs(n=240, seed=11, p_a=0.8, p_b=0.6) -> Pairs:
    rng = np.random.default_rng(seed)
    a = (rng.random(n) < p_a).astype(float)
    b = (rng.random(n) < p_b).astype(float)
    return Pairs("a", "b", tuple(f"q{i}" for i in range(n)), a, b)


def test_simulate_reports_error_rates_power_and_items():
    report = simulate(effects=(0.0, 0.15), items=300, runs=1500, discordance=0.3, seed=1)
    rows = {(row.effect, row.rule): row for row in report.rows}
    assert len(rows) == 2 * len(rules.RULES)
    limit = 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / 1500)
    # Tolerance: three binomial standard errors above alpha.
    for name in ("betting", "mixture", "discordant", "fixed"):
        assert rows[(0.0, name)].wrong <= limit
        assert rows[(0.0, name)].right == 0.0
    assert rows[(0.0, "naive")].wrong > 0.15
    assert rows[(0.15, "discordant")].right > 0.8
    assert rows[(0.15, "discordant")].mean_items < rows[(0.15, "fixed")].mean_items == 300.0
    assert rows[(0.15, "fixed")].right >= rows[(0.15, "betting")].right
    assert all(row.futile == 0.0 for row in report.rows)


def test_simulate_is_reproducible_and_chunking_does_not_change_it(monkeypatch):
    kwargs = dict(effects=(0.1,), items=120, runs=300, rule_names=("discordant", "naive"))
    first = simulate(seed=5, **kwargs)
    assert simulate(seed=5, **kwargs).rows == first.rows
    assert simulate(seed=6, **kwargs).rows != first.rows
    monkeypatch.setattr("evalstop.simulate._CHUNK_CELLS", 120 * 70)
    chunked = simulate(seed=5, **kwargs)
    assert [row.rule for row in chunked.rows] == ["discordant", "naive"]
    assert chunked.rows[0].right == pytest.approx(first.rows[0].right, abs=0.15)


def test_simulate_futility_and_negative_effects():
    report = simulate(
        effects=(0.0, -0.3),
        items=800,
        runs=300,
        discordance=0.4,
        min_effect=0.2,
        rule_names=("betting",),
        seed=2,
    )
    null, negative = report.rows
    assert null.futile > 0.5
    assert negative.right > 0.9 and negative.futile <= 0.05
    text = report.to_markdown()
    assert "futility below a difference of 0.2" in text
    assert "| -0.3 | betting | yes |" in text
    data = report.to_dict()
    assert data["rows"][0]["power"] is None
    assert data["rows"][1]["power"] == pytest.approx(negative.right, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rule_names": ("exact",)}, "unknown rule"),
        ({"rule_names": ()}, "at least one rule"),
        ({"items": 0}, "items and runs"),
        ({"discordance": 0.0}, "discordance"),
        ({"effects": (0.5,), "discordance": 0.3}, "larger than the discordance"),
        ({"alpha": 2.0}, "alpha"),
    ],
)
def test_simulate_rejects_bad_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate(**{"runs": 10, "items": 20, **kwargs})


def test_replay_reports_items_and_agreement():
    pairs = make_pairs()
    report = replay(pairs, orders=400, seed=3, source="memory")
    diff = pairs.diff
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    assert (report.wins, report.losses) == (wins, losses)
    assert report.full_p == pytest.approx(mcnemar_exact_p(wins, losses))
    assert report.full_decision == "a"
    rows = {row.rule: row for row in report.rows}
    assert rows["fixed"].decided == rows["fixed"].agree == 1.0
    assert rows["fixed"].mean_share == 1.0 and rows["fixed"].median_items == 240.0
    for name in ("betting", "discordant", "naive"):
        # Deciding for A is agreeing with the full-data test here.
        assert rows[name].agree <= rows[name].decided + 1e-12
        assert rows[name].mean_share < 1.0
    assert rows["discordant"].decided > 0.7
    assert rows["naive"].mean_share < rows["discordant"].mean_share
    text = report.to_markdown()
    assert "exact McNemar test on all pairs: p = " in text and "`a` better" in text
    assert "| discordant | yes |" in text
    assert report.to_dict()["rows"][0]["agrees_with_full_data"] is not None


def test_replay_null_measures_false_positives():
    pairs = make_pairs(n=300)
    report = replay(pairs, orders=1500, null=True, seed=4)
    rows = {row.rule: row for row in report.rows}
    limit = 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / 1500)
    # Tolerance: three binomial standard errors above alpha.
    for name in ("betting", "mixture", "discordant", "fixed"):
        assert rows[name].decided <= limit
    assert rows["naive"].decided > 0.15
    text = report.to_markdown()
    assert "labels swapped at random within each pair (null)" in text
    assert "| rule | valid at every look | false-positive rate |" in text
    assert report.to_dict()["rows"][0]["agrees_with_full_data"] is None


def test_replay_when_the_full_data_test_is_not_significant():
    pairs = make_pairs(n=200, p_a=0.6, p_b=0.6, seed=12)
    report = replay(pairs, orders=300, seed=5, min_effect=0.3, rule_names=("betting", "fixed"))
    assert report.full_decision is None
    rows = {row.rule: row for row in report.rows}
    assert rows["fixed"].agree == 1.0 and rows["fixed"].decided == 0.0
    assert rows["betting"].agree == pytest.approx(1.0 - rows["betting"].decided)
    assert "no significant difference" in report.to_markdown()
    assert "futility below a difference of 0.3" in report.to_markdown()


def test_replay_skips_mcnemar_rules_for_graded_scores():
    rng = np.random.default_rng(8)
    a, b = rng.random(150), rng.random(150) * 0.6
    pairs = Pairs("a", "b", tuple(str(i) for i in range(150)), a, b)
    report = replay(pairs, orders=50, seed=1)
    assert report.skipped == ("discordant", "naive", "fixed")
    assert [row.rule for row in report.rows] == ["betting", "mixture"]
    assert "skipped (scores are not pass/fail): discordant, naive, fixed" in report.to_markdown()
    with pytest.raises(ValueError, match="need pass/fail scores"):
        replay(pairs, orders=10, rule_names=("naive",))


def test_replay_rejects_bad_input():
    empty = Pairs("a", "b", (), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="no item"):
        replay(empty)
    with pytest.raises(ValueError, match="orders"):
        replay(make_pairs(n=10), orders=0)
    wide = Pairs("a", "b", ("x",), np.array([5.0]), np.array([1.0]))
    with pytest.raises(ValueError, match=r"scores in \[0, 1\]"):
        replay(wide)


def test_simulate_and_replay_commands(tmp_path, capsys):
    assert main(["simulate", "--effects", "0", "0.2", "--items", "150", "--runs", "200",
                 "--rules", "discordant", "naive", "--look-every", "10",
                 "--format", "json"]) == 0  # fmt: skip
    data = json.loads(capsys.readouterr().out)
    assert data["command"] == "simulate" and data["every"] == 10
    assert [(row["effect"], row["rule"]) for row in data["rows"]] == [
        (0.0, "discordant"),
        (0.0, "naive"),
        (0.2, "discordant"),
        (0.2, "naive"),
    ]
    assert main(["simulate", "--items", "100", "--runs", "100", "--look-every", "5"]) == 0
    assert "naive looks every 5 items" in capsys.readouterr().out

    pairs = make_pairs(n=120)
    rows = []
    for item, x, y in zip(pairs.items, pairs.a, pairs.b, strict=True):
        rows.append({"item": item, "model": "a", "score": int(x)})
        rows.append({"item": item, "model": "b", "score": int(y)})
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert main(["replay", str(path), "--orders", "100", "--seed", "9", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["command"] == "replay" and data["orders"] == 100 and data["pairs"] == 120
    assert main(["replay", str(path), "--orders", "100", "--null"]) == 0
    assert "false-positive rate" in capsys.readouterr().out
    assert main(["replay", str(path), "--orders", "0"]) == 2
    assert "orders must be at least 1" in capsys.readouterr().err
