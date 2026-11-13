"""Write examples/results.jsonl: simulated per-item results for two models.

The data are synthetic, not taken from any real evaluation. Each item has a
latent difficulty shared by both models, so their results are correlated the
way results on a shared benchmark usually are. Rows are in evaluation order:
item by item, both models per item. The seed was picked so that the realised
pass rates (0.72 and 0.62) sit close to the nominal ones below.

Run from the repository root:

    python examples/make_example.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

SEED = 5
ITEMS = 300
PASS_RATES = {"model-a": 0.72, "model-b": 0.60}
# Share of the latent score that comes from the item rather than the model.
SHARED = 0.6


def simulate(seed: int = SEED, items: int = ITEMS) -> list[dict[str, object]]:
    """Return the rows of the example file."""
    rng = np.random.default_rng(seed)
    difficulty = rng.standard_normal(items)
    noise = {model: rng.standard_normal(items) for model in PASS_RATES}
    cutoff = {model: NormalDist().inv_cdf(rate) for model, rate in PASS_RATES.items()}
    rows: list[dict[str, object]] = []
    for i in range(items):
        for model in PASS_RATES:
            latent = math.sqrt(SHARED) * difficulty[i] + math.sqrt(1 - SHARED) * noise[model][i]
            rows.append(
                {"item": f"item-{i + 1:03d}", "model": model, "score": int(latent < cutoff[model])}
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write simulated per-item results for two models (not real data)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="file to write (default: results.jsonl next to this script)",
    )
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else Path(__file__).with_name("results.jsonl")
    out.write_text("".join(json.dumps(row) + "\n" for row in simulate()), encoding="utf-8")
    print(f"wrote {out.name}: {ITEMS} items, {len(PASS_RATES)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
