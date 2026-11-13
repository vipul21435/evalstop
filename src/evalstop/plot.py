"""Optional chart of a confidence sequence (needs matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evalstop.sequence import ConfidenceSequence


def _pyplot():
    try:
        import matplotlib
    except ImportError:
        raise ValueError(
            "the chart needs matplotlib; install it with: pip install 'evalstop[plot]'"
        ) from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_sequence(
    sequence: ConfidenceSequence,
    path: str | Path,
    *,
    title: str = "",
    ylabel: str = "mean score",
    reference: float | None = None,
    stops: Sequence[tuple[int, str]] = (),
) -> Path:
    """Draw the running mean and the confidence sequence and save the figure.

    ``reference`` draws a horizontal line (a threshold, or 0 for a difference).
    ``stops`` is a list of ``(items, label)`` pairs marked with vertical lines.
    """
    if len(sequence) == 0:
        raise ValueError("nothing to draw: the sequence is empty")
    plt = _pyplot()
    t = np.arange(1, len(sequence) + 1)
    level = f"{100 * (1 - sequence.alpha):g}%"
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    ax.fill_between(
        t, sequence.lower, sequence.upper, color="#9ecae1", alpha=0.6, label=f"{level} sequence"
    )
    ax.plot(t, sequence.mean, color="#08519c", linewidth=1.2, label="running mean")
    if reference is not None:
        ax.axhline(reference, color="#444444", linewidth=0.8, linestyle="--")
    colors = ["#d94801", "#238b45", "#6a51a3", "#525252"]
    for i, (n, label) in enumerate(stops):
        ax.axvline(n, color=colors[i % len(colors)], linewidth=1.0, linestyle=":", label=label)
    ax.set_xlabel("items seen")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, len(sequence))
    ax.set_ylim(*sequence.bounds)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out)
    plt.close(fig)
    return out
