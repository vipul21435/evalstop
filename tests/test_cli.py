"""The monitor command and its reports."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalstop import betting_cs, mixture_cs
from evalstop.cli import main
from evalstop.monitor import monitor_scores

REPO = Path(__file__).resolve().parents[1]


def make_scores(n: int = 240, rate: float = 0.8, seed: int = 3) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.random(n) < rate]


def write_jsonl(path: Path, scores, model: str = "m") -> Path:
    rows = [{"item": f"q{i}", "model": model, "score": score} for i, score in enumerate(scores)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    status = main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_json_report_matches_the_library(tmp_path, capsys):
    scores = make_scores()
    path = write_jsonl(tmp_path / "run.jsonl", scores)
    status, out, err = run(
        capsys, "monitor", str(path), "--threshold", "0.5", "--width", "0.3", "--format", "json"
    )
    assert status == 0 and err == ""
    report = json.loads(out)
    seq = betting_cs(scores, 0.05)
    assert report["command"] == "monitor"
    assert report["source"] == str(path)
    assert report["model"] == "m"
    assert report["method"] == "betting"
    assert report["alpha"] == 0.05
    assert report["settings"] == {"grid": 1000}
    assert report["n"] == 240
    assert report["mean"] == pytest.approx(sum(scores) / 240, abs=1e-6)
    assert report["lower"] == pytest.approx(seq.lower[-1], abs=1e-6)
    assert report["upper"] == pytest.approx(seq.upper[-1], abs=1e-6)
    assert report["first_empty"] is None
    threshold, width = report["rules"]
    stop = seq.threshold_stop(0.5)
    assert threshold == {
        "rule": "threshold",
        "value": 0.5,
        "stopped_at": stop.n,
        "decision": "above",
        "lower": pytest.approx(stop.lower, abs=1e-6),
        "upper": pytest.approx(stop.upper, abs=1e-6),
    }
    assert width["rule"] == "width" and width["stopped_at"] == seq.width_stop(0.3).n
    assert [row["n"] for row in report["trace"]] == list(range(24, 241, 24))
    assert report["trace"][-1]["lower"] == report["lower"]


def test_markdown_report(tmp_path, capsys):
    scores = make_scores()
    path = write_jsonl(tmp_path / "run.jsonl", scores)
    status, out, _ = run(capsys, "monitor", str(path), "--threshold", "0.5", "--width", "0.01")
    assert status == 0
    seq = betting_cs(scores, 0.05)
    stop = seq.threshold_stop(0.5)
    assert out.startswith("# evalstop monitor\n")
    assert f"- input: `{path}`, model `m`" in out
    assert "- scores: 240 in [0, 1], mean " in out
    assert "- method: betting (hedged capital), alpha 0.05, grid 1000" in out
    assert (
        f"- 95% confidence sequence after 240 scores: [{seq.lower[-1]:.4f}, {seq.upper[-1]:.4f}]"
        in out
    )
    assert f"| threshold 0.5 | {stop.n} of 240 scores | mean above 0.5 |" in out
    assert "| width <= 0.01 | not within 240 scores | not reached | |" in out
    assert out.count("\n| ") == 2 + 2 + 10
    assert "Warning" not in out


def test_no_rules_means_no_rules_section(tmp_path, capsys):
    path = write_jsonl(tmp_path / "run.jsonl", make_scores(40))
    status, out, _ = run(capsys, "monitor", str(path))
    assert status == 0
    assert "## Stopping rules" not in out
    assert "## Trace" in out


def test_mixture_method_and_trace_spacing(tmp_path, capsys):
    scores = make_scores(100, rate=0.3)
    path = write_jsonl(tmp_path / "run.jsonl", scores)
    status, out, _ = run(
        capsys,
        "monitor",
        str(path),
        "--method",
        "mixture",
        "--tune-at",
        "50",
        "--alpha",
        "0.1",
        "--every",
        "30",
        "--threshold",
        "0.9",
        "--format",
        "json",
    )
    assert status == 0
    report = json.loads(out)
    seq = mixture_cs(scores, 0.1, tune_at=50)
    assert report["method"] == "mixture"
    assert report["settings"] == {"tune_at": 50.0}
    assert report["upper"] == pytest.approx(seq.upper[-1], abs=1e-6)
    assert [row["n"] for row in report["trace"]] == [30, 60, 90, 100]
    assert report["rules"][0]["decision"] == "below"


def test_csv_on_standard_input_with_a_model_choice(monkeypatch, capsys):
    text = "item,model,score\nq1,a,1\nq1,b,0\nq2,a,1\nq2,b,1\nq3,a,0\nq3,b,0\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    status, out, _ = run(capsys, "monitor", "-", "--model", "b", "--format", "json")
    assert status == 0
    report = json.loads(out)
    assert report["source"] == "standard input"
    assert report["model"] == "b"
    assert report["n"] == 3
    assert report["mean"] == pytest.approx(1 / 3, abs=1e-6)


def test_other_score_ranges(tmp_path, capsys):
    path = write_jsonl(tmp_path / "run.jsonl", [3, 7, 10, 0, 6, 8, 9, 5])
    status, out, _ = run(
        capsys, "monitor", str(path), "--lo", "0", "--hi", "10", "--grid", "100", "--format", "json"
    )
    assert status == 0
    report = json.loads(out)
    assert report["bounds"] == [0.0, 10.0]
    assert report["mean"] == 6.0
    assert 0.0 <= report["lower"] <= 6.0 <= report["upper"] <= 10.0


def test_a_change_of_level_is_flagged(tmp_path, capsys):
    path = write_jsonl(tmp_path / "run.jsonl", [0] * 300 + [1] * 300)
    status, out, _ = run(capsys, "monitor", str(path), "--grid", "200")
    assert status == 0
    assert "Warning: after " in out
    status, out, _ = run(capsys, "monitor", str(path), "--grid", "200", "--format", "json")
    assert json.loads(out)["first_empty"] > 300


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "several models"),
        (["--model", "c"], "no results for model 'c'"),
        (["--model", "a", "--alpha", "1.5"], "alpha"),
        (["--model", "a", "--threshold", "1"], "threshold"),
        (["--model", "a", "--width", "0"], "width"),
        (["--model", "a", "--every", "0"], "every"),
        (["--model", "a", "--grid", "1"], "grid"),
        (["--model", "a", "--hi", "0.5"], "outside"),
        (["--model", "a", "--method", "mixture", "--hi", "0.5"], r"scores must lie in \[0, 0.5\]"),
    ],
)
def test_bad_requests_exit_with_status_2(tmp_path, capsys, extra, message):
    rows = [{"item": "q1", "model": "a", "score": 1}, {"item": "q1", "model": "b", "score": 0}]
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    status, out, err = run(capsys, "monitor", str(path), *extra)
    assert status == 2
    assert out == ""
    assert err.startswith("evalstop: error: ")
    assert re.search(message, err)


def test_missing_and_empty_files_exit_with_status_2(tmp_path, capsys):
    status, _, err = run(capsys, "monitor", str(tmp_path / "absent.jsonl"))
    assert status == 2 and "absent.jsonl" in err
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    status, _, err = run(capsys, "monitor", str(empty))
    assert status == 2 and "no scores to monitor" in err
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"score": 1}\n{"score": \n', encoding="utf-8")
    status, _, err = run(capsys, "monitor", str(broken))
    assert status == 2 and "line 2: not valid JSON" in err


def test_monitor_rejects_a_repeated_item_like_compare_does(tmp_path, capsys):
    rows = [
        {"item": "q1", "model": "a", "score": 1},
        {"item": "q1", "model": "b", "score": 0},
        {"item": "q1", "model": "a", "score": 0},
    ]
    path = tmp_path / "repeat.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for argv in (["monitor", str(path), "--model", "a"], ["compare", str(path)]):
        status, out, err = run(capsys, *argv)
        assert status == 2 and out == ""
        assert "model 'a' has two scores for item 'q1'" in err
    # Rows without an item field are numbered, so they never collide.
    plain = tmp_path / "plain.csv"
    plain.write_text("score\n1\n1\n0\n", encoding="utf-8")
    assert run(capsys, "monitor", str(plain))[0] == 0


def test_unknown_method_is_rejected_by_the_library():
    with pytest.raises(ValueError, match="unknown method"):
        monitor_scores([0, 1], method="exact")


def test_usage_errors_come_from_argparse(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("evalstop ")


def test_the_example_file_runs(capsys):
    example = REPO / "examples" / "results.jsonl"
    status, out, _ = run(capsys, "monitor", str(example), "--model", "model-a", "--format", "json")
    assert status == 0
    report = json.loads(out)
    assert report["n"] == 300
    assert report["lower"] < report["mean"] < report["upper"]


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "evalstop", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("evalstop ")
