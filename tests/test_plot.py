"""The optional chart."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from evalstop import betting_cs, mixture_cs
from evalstop.cli import main
from evalstop.plot import plot_sequence

PNG = b"\x89PNG\r\n\x1a\n"


def test_plot_writes_a_png(tmp_path):
    rng = np.random.default_rng(1)
    seq = betting_cs(rng.random(120) < 0.7, 0.05, grid=200)
    out = plot_sequence(
        seq, tmp_path / "seq.png", title="demo", reference=0.5, stops=[(40, "threshold")]
    )
    data = out.read_bytes()
    assert data.startswith(PNG) and len(data) > 5000


def test_plot_rejects_an_empty_sequence(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        plot_sequence(mixture_cs([]), tmp_path / "none.png")


def test_missing_matplotlib_is_explained(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ValueError, match=r"pip install 'evalstop\[plot\]'"):
        plot_sequence(mixture_cs([0.0, 1.0]), tmp_path / "none.png")


def test_chart_option_of_monitor_and_compare(tmp_path, capsys):
    rng = np.random.default_rng(2)
    rows = []
    for i in range(150):
        rows.append({"item": f"q{i}", "model": "a", "score": int(rng.random() < 0.85)})
        rows.append({"item": f"q{i}", "model": "b", "score": int(rng.random() < 0.45)})
    path = tmp_path / "two.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    one = tmp_path / "monitor.png"
    assert main(["monitor", str(path), "--model", "a", "--threshold", "0.5",
                 "--chart", str(one)]) == 0  # fmt: skip
    two = tmp_path / "compare.png"
    assert main(["compare", str(path), "--chart", str(two), "--method", "mixture"]) == 0
    capsys.readouterr()
    assert one.read_bytes().startswith(PNG) and two.read_bytes().startswith(PNG)
