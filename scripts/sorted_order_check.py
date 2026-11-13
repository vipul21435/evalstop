"""Show what a difficulty-sorted order does to a confidence sequence.

Every item passes with its own probability, falling in a straight line from
0.8 for the first item to 0.2 for the last. The mean over the pool is 0.5, but
the easy items come first. The sequences assume that every score has the same
expected value given the earlier ones, and such an order breaks that.

    python scripts/sorted_order_check.py --runs 1000 --seed 0

Prints, over the runs, how often the betting sequence excludes the true mean
at some look and how often its running intersection becomes empty (the only
symptom ``evalstop monitor`` warns about), for the sorted order and for the
same scores shuffled.
"""

from __future__ import annotations

import argparse

import numpy as np

from evalstop import betting_cs


def measure(runs: int, items: int, alpha: float, seed: int, grid: int = 200) -> dict[str, float]:
    """Shares of runs that miss the pool mean and that end with an empty intersection."""
    rng = np.random.default_rng(seed)
    rates = np.linspace(0.8, 0.2, items)
    counts = {"sorted_miss": 0, "sorted_empty": 0, "shuffled_miss": 0, "shuffled_empty": 0}
    for _ in range(runs):
        scores = (rng.random(items) < rates).astype(float)
        for name, ordered in (("sorted", scores), ("shuffled", rng.permutation(scores))):
            seq = betting_cs(ordered, alpha, grid=grid)
            counts[f"{name}_miss"] += seq.threshold_stop(0.5) is not None
            counts[f"{name}_empty"] += seq.first_empty is not None
    return {key: value / runs for key, value in counts.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    shares = measure(args.runs, args.items, args.alpha, args.seed)
    print(
        f"{args.runs} runs of {args.items} items, pass probability 0.8 down to 0.2 "
        f"(pool mean 0.5), alpha {args.alpha:g}, seed {args.seed}"
    )
    for name, label in (("sorted", "sorted by difficulty"), ("shuffled", "same scores shuffled")):
        print(
            f"{label}: true mean excluded in {100 * shares[f'{name}_miss']:.1f} % of runs, "
            f"empty-intersection warning in {100 * shares[f'{name}_empty']:.1f} %"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
