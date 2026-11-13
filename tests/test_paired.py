"""Pairing two models and the compare command."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evalstop import betting_cs, mixture_cs, rules
from evalstop.cli import main
from evalstop.io import InputError, Record
from evalstop.paired import Pairs, compare_pairs, pair_records
from evalstop.stats import mcnemar_exact_p


def records_for(a, b, name_a="a", name_b="b"):
    out = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        out.append(Record(f"q{i}", name_a, float(x)))
        out.append(Record(f"q{i}", name_b, float(y)))
    return out


def simulated(n=300, seed=4, p_a=0.85, p_b=0.5):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p_a).astype(int), (rng.random(n) < p_b).astype(int)


def write(path, records):
    rows = [{"item": r.item, "model": r.model, "score": r.score} for r in records]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pairs_follow_the_order_in_which_they_become_complete():
    records = [
        Record("q1", "a", 1.0),
        Record("q2", "a", 0.0),
        Record("q3", "a", 1.0),
        Record("q9", "c", 1.0),
        Record("q2", "b", 1.0),
        Record("q1", "b", 0.0),
        Record("q4", "b", 1.0),
    ]
    pairs = pair_records(records, "a", "b")
    assert pairs.items == ("q2", "q1")
    assert pairs.a.tolist() == [0.0, 1.0]
    assert pairs.b.tolist() == [1.0, 0.0]
    assert pairs.diff.tolist() == [-1.0, 1.0]
    assert pairs.unpaired == 2
    assert len(pairs) == 2 and pairs.binary
    swapped = pair_records(records, "b", "a")
    assert swapped.diff.tolist() == [1.0, -1.0]


def test_two_models_are_found_without_naming_them():
    pairs = pair_records(records_for([1, 0], [0.5, 0.25], "left", "right"))
    assert (pairs.model_a, pairs.model_b) == ("left", "right")
    assert not pairs.binary


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "need two models to compare; found 'a', 'b', 'c'"),
        ({"model_a": "a"}, "name both models"),
        ({"model_a": "a", "model_b": "a"}, "must differ"),
        ({"model_a": "a", "model_b": "z"}, "no results for model 'z'"),
    ],
)
def test_pairing_errors(kwargs, message):
    records = [*records_for([1], [0]), Record("q0", "c", 1.0)]
    with pytest.raises(InputError, match=message):
        pair_records(records, **kwargs)


def test_a_repeated_item_is_an_error():
    records = [*records_for([1, 0], [0, 0]), Record("q1", "b", 1.0)]
    with pytest.raises(InputError, match="two scores for item 'q1'"):
        pair_records(records)


def test_compare_lines_match_the_library():
    a, b = simulated()
    pairs = pair_records(records_for(a, b))
    report = compare_pairs(pairs, alpha=0.05, min_effect=0.05, source="memory")
    diff = (a - b).astype(float)
    seq = betting_cs(diff, 0.05, lo=-1.0, hi=1.0)
    np.testing.assert_allclose(report.sequence.lower, seq.lower)
    by_name = {line.name: line for line in report.lines}
    assert list(by_name) == ["betting", "discordant", "naive", "fixed"]
    settled = seq.threshold_stop(0.0)
    assert (by_name["betting"].stopped_at, by_name["betting"].decision) == (settled.n, "a")
    for name in ("discordant", "naive", "fixed"):
        outcome = rules.run_rule(name, diff, 0.05)
        assert by_name[name].decision == "a"
        assert by_name[name].stopped_at == int(outcome.stop[0])
    assert by_name["naive"].stopped_at <= by_name["discordant"].stopped_at
    assert [line.valid for line in report.lines] == [True, True, False, False]
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    assert f"p = {mcnemar_exact_p(wins, losses):.3g}" in by_name["fixed"].note

    data = report.to_dict()
    assert data["command"] == "compare"
    assert data["pairs"] == 300 and data["unpaired"] == 0
    assert data["difference"] == pytest.approx(diff.mean(), abs=1e-6)
    assert data["counts"]["a_only"] == wins and data["counts"]["b_only"] == losses
    assert sum(data["counts"].values()) == 300
    assert [rule["rule"] for rule in data["rules"]] == list(by_name)
    assert len(data["trace"]) == 10

    text = report.to_markdown()
    assert text.startswith("# evalstop compare\n")
    assert "| betting sequence on the difference | yes |" in text
    assert "| exact McNemar at every look | no |" in text
    assert "`a` better" in text


def test_futility_and_undecided_outcomes():
    rng = np.random.default_rng(6)
    same = (rng.random(1500) < 0.5).astype(int)
    flips = rng.random(1500) < 0.08
    other = np.where(flips, 1 - same, same)
    pairs = pair_records(records_for(same, other))
    futile = compare_pairs(pairs, min_effect=0.15)
    first = futile.lines[0]
    assert first.decision == "futile" and first.stopped_at < 1500
    assert "no difference of 0.15 or more" in futile.to_markdown()
    open_ = compare_pairs(pairs)
    assert open_.lines[0].decision is None and open_.lines[0].stopped_at is None
    assert "not within 1500 pairs | undecided" in open_.to_markdown()


def test_graded_scores_use_only_the_sequence():
    rng = np.random.default_rng(7)
    a = np.round(rng.uniform(4.0, 10.0, 200), 1)
    b = np.round(rng.uniform(0.0, 7.0, 200), 1)
    pairs = Pairs("a", "b", tuple(f"q{i}" for i in range(200)), a, b)
    report = compare_pairs(pairs, method="mixture", lo=0.0, hi=10.0, tune_at=50)
    assert [line.name for line in report.lines] == ["mixture"]
    assert report.lines[0].decision == "a"
    seq = mixture_cs(a - b, 0.05, lo=-10.0, hi=10.0, tune_at=50)
    np.testing.assert_allclose(report.sequence.upper, seq.upper)
    assert report.to_dict()["counts"] is None
    assert "mixture sequence on the difference" in report.to_markdown()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "exact"}, "unknown method"),
        ({"min_effect": 1.0}, "min_effect"),
        ({"hi": 0.5}, r"scores must lie in \[0, 0.5\]"),
        ({"every": 0}, "every"),
    ],
)
def test_compare_rejects_bad_settings(kwargs, message):
    pairs = pair_records(records_for([1, 0, 1], [0, 0, 1]))
    with pytest.raises(ValueError, match=message):
        compare_pairs(pairs, **kwargs)


def test_compare_needs_at_least_one_pair():
    records = [Record("q1", "a", 1.0), Record("q2", "b", 0.0)]
    with pytest.raises(ValueError, match="no item has a score from both models"):
        compare_pairs(pair_records(records))


def test_compare_command(tmp_path, capsys):
    a, b = simulated()
    records = [*records_for(a, b, "strong", "weak"), Record("extra", "strong", 1.0)]
    path = write(tmp_path / "two.jsonl", records)
    assert main(["compare", str(path), "--min-effect", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "- A: `strong`, mean " in out and "B: `weak`, mean " in out
    assert "- left out: 1 score for items only one model was run on" in out
    assert main(["compare", str(path), "--model-a", "weak", "--model-b", "strong",
                 "--method", "mixture", "--format", "json"]) == 0  # fmt: skip
    data = json.loads(capsys.readouterr().out)
    assert data["model_a"] == "weak" and data["difference"] < 0
    assert data["method"] == "mixture"
    assert {rule["rule"]: rule["decision"] for rule in data["rules"]}["fixed"] == "b"
    assert main(["compare", str(path), "--model-a", "weak"]) == 2
    assert "name both models" in capsys.readouterr().err
