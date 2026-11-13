"""The fetch and study scripts, run offline on made-up data."""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
EXAMPLES = SCRIPTS.parent / "examples"


def load(name: str, folder: Path = SCRIPTS):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def not_found(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 404, "Not Found", None, None)


def test_fetch_reads_both_layouts_without_the_network(tmp_path, monkeypatch, capsys):
    fetch = load("fetch_swebench_verified")

    def fake(url: str):
        if url.endswith("new/per_instance_details.json"):
            return {"b": {"resolved": True}, "a": {"resolved": False}, "c": {"resolved": True}}
        if url.endswith("old/results/results.json"):
            return {"resolved": ["a"], "no_generation": ["c"], "note": "text"}
        raise not_found(url)

    monkeypatch.setattr(fetch, "fetch_json", fake)
    out = tmp_path / "verified.jsonl"
    assert fetch.main(["--out", str(out), "--submissions", "new", "old"]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"item": "a", "model": "new", "score": 0},
        {"item": "a", "model": "old", "score": 1},
        {"item": "b", "model": "new", "score": 1},
        {"item": "b", "model": "old", "score": 0},
        {"item": "c", "model": "new", "score": 1},
        {"item": "c", "model": "old", "score": 0},
    ]
    assert "3 items, 2 submissions" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no per-instance results in a known layout"):
        fetch.main(["--out", str(out), "--submissions", "missing"])


def test_study_prints_its_tables_and_writes_the_chart(tmp_path, capsys):
    study = load("swebench_study")
    rng = np.random.default_rng(3)
    models = sorted({name for pair in study.PAIRS for name in pair})
    rates = dict(zip(models, np.linspace(0.45, 0.8, len(models)), strict=True))
    rows = []
    for i in range(120):
        for model in models:
            score = int(rng.random() < rates[model])
            rows.append({"item": f"q{i}", "model": study.PREFIX + model, "score": score})
    data = tmp_path / "made_up.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dump, chart = tmp_path / "study.json", tmp_path / "study.png"
    args = [str(data), "--orders", "40", "--json", str(dump), "--chart", str(chart)]
    assert study.main(args) == 0
    out = capsys.readouterr().out
    assert "Mean share of the 500 items used" in out
    assert "False-positive rate with labels swapped" in out
    assert out.count("\n| ") == 4 * (1 + len(study.PAIRS))
    saved = json.loads(dump.read_text(encoding="utf-8"))
    assert len(saved) == len(study.PAIRS) and saved[0]["null"]["null"] is True
    assert chart.read_bytes().startswith(b"\x89PNG")


def test_example_generator_takes_an_output_path_and_prints_help(tmp_path, capsys):
    make = load("make_example", EXAMPLES)
    committed = (EXAMPLES / "results.jsonl").read_bytes()
    out = tmp_path / "made.jsonl"
    with pytest.raises(SystemExit) as excinfo:
        make.main(["--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out and not out.exists()
    with pytest.raises(SystemExit) as excinfo:
        make.main(["--out", str(out), "--help"])
    assert excinfo.value.code == 0 and not out.exists()
    capsys.readouterr()

    assert make.main(["--out", str(out)]) == 0
    assert "wrote made.jsonl: 300 items, 2 models" in capsys.readouterr().out
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == make.simulate()
    assert len(rows) == 600 and {row["model"] for row in rows} == {"model-a", "model-b"}
    # Neither call touched the committed example.
    assert (EXAMPLES / "results.jsonl").read_bytes() == committed
