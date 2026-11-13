"""Watch a comparison item by item and stop as soon as it is settled.

Uses the simulated results in ``examples/results.jsonl`` (not real data).
Run from the repository root:

    python examples/stop_early.py
"""

from __future__ import annotations

from pathlib import Path

from evalstop import BettingCS, pair_records
from evalstop.io import read_records


def main() -> None:
    records = read_records(Path(__file__).with_name("results.jsonl"))
    pairs = pair_records(records, "model-a", "model-b")

    # A score difference lies in [-1, 1]; the sequence covers its mean at every look.
    sequence = BettingCS(alpha=0.05, lo=-1.0, hi=1.0)
    for seen, difference in enumerate(pairs.diff, start=1):
        lower, upper = sequence.update(difference)
        if lower >= 0.0 or upper <= 0.0:
            better = pairs.model_a if lower >= 0.0 else pairs.model_b
            print(f"stopped after {seen} of {len(pairs)} items: {better} is better")
            interval = f"[{lower:+.3f}, {upper:+.3f}]"
            print(f"mean difference so far {sequence.mean:+.3f}, interval {interval}")
            break
    else:
        interval = f"[{sequence.lower:+.3f}, {sequence.upper:+.3f}]"
        print(f"still open after all {len(pairs)} items: {interval}")


if __name__ == "__main__":
    main()
