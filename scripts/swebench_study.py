"""Replay the stopping rules over pairs of public SWE-bench Verified submissions.

    python scripts/fetch_swebench_verified.py --out verified.jsonl
    python scripts/swebench_study.py verified.jsonl --orders 2000 --seed 0

Prints Markdown tables: how many of the 500 items each rule needs before it
decides when the real results arrive in a random order, how often its decision
matches the exact McNemar test on all 500, and the false-positive rate of each
rule when the two labels are swapped at random within each pair (which makes
"no difference" true by construction).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalstop import rules
from evalstop.io import read_records
from evalstop.paired import pair_records
from evalstop.simulate import replay

PREFIX = "20260217_mini-v2.0.0_"
# Pairs of submissions with one scaffold, from a large gap in resolved items to none.
PAIRS = (
    ("gemini-3-flash-high", "gpt-5-mini"),
    ("kimi-k2-5-high", "gpt-5-mini"),
    ("gemini-3-flash-high", "deepseek-3-2-high"),
    ("minimax-2-5-high", "kimi-k2-5-high"),
    ("glm-5-high", "deepseek-3-2-high"),
    ("gpt-5-2-high", "glm-5-high"),
)


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


SHOWN = ("betting", "discordant", "sprt", "obf", "pocock", "naive")


def draw(dump: list[dict], gaps: list[str], alpha: float, path: Path):
    """Items used by each rule per pair, and false-positive rates under the null.

    Saves the figure to ``path`` and returns it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=120)
    x = range(len(dump))
    for name in SHOWN:
        share = [
            next(r["mean_share_of_items"] for r in entry["real"]["rows"] if r["rule"] == name)
            for entry in dump
        ]
        style = "--" if name == "naive" else "-"
        left.plot(x, share, marker="o", linestyle=style, linewidth=1.2, label=name)
    left.set_xticks(list(x), gaps)
    left.set_xlabel("gap between the two submissions (points)")
    left.set_ylabel("mean share of the 500 items used")
    left.set_ylim(0, 1.05)
    left.legend(fontsize=8)
    left.set_title("Real results in random orders", fontsize=10)

    rates = {
        name: [
            next(r["decided"] for r in entry["null"]["rows"] if r["rule"] == name)
            for entry in dump
        ]
        for name in SHOWN
    }
    means = [sum(rates[name]) / len(dump) for name in SHOWN]
    low = [m - min(rates[name]) for m, name in zip(means, SHOWN, strict=True)]
    high = [max(rates[name]) - m for m, name in zip(means, SHOWN, strict=True)]
    colors = ["#d94801" if name == "naive" else "#4292c6" for name in SHOWN]
    right.bar(list(SHOWN), means, yerr=[low, high], color=colors, capsize=3)
    right.axhline(alpha, color="#444444", linewidth=0.8, linestyle="--")
    # Slanted so that neighbouring rule names do not run together.
    right.tick_params(axis="x", labelrotation=30)
    right.set_ylabel("false-positive rate")
    right.set_title("Labels swapped at random (null); bars span the six pairs", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="file written by fetch_swebench_verified.py")
    parser.add_argument("--orders", type=int, default=2000, help="random orders per pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--json", type=Path, help="also write every number to this JSON file")
    parser.add_argument("--chart", type=Path, help="also save a chart of the results (PNG)")
    args = parser.parse_args(argv)

    records = read_records(args.results)
    names = list(rules.RULES)
    summary, shares, agreement, false_positives, dump = [], [], [], [], []
    for a, b in PAIRS:
        pairs = pair_records(records, PREFIX + a, PREFIX + b)
        real = replay(pairs, orders=args.orders, alpha=args.alpha, seed=args.seed)
        null = replay(pairs, orders=args.orders, alpha=args.alpha, seed=args.seed, null=True)
        label = f"{a} vs {b}"
        gap = 100.0 * (pairs.a.mean() - pairs.b.mean())
        summary.append(
            [
                label,
                f"{int(pairs.a.sum())}",
                f"{int(pairs.b.sum())}",
                f"{gap:+.1f}",
                f"{real.wins}",
                f"{real.losses}",
                f"{real.full_p:.2g}",
            ]
        )
        by_rule = {row.rule: row for row in real.rows}
        by_rule_null = {row.rule: row for row in null.rows}
        shares.append(
            [label]
            + [f"{by_rule[n].mean_share:.2f} ({100 * by_rule[n].decided:.0f}%)" for n in names]
        )
        agreement.append([label] + [f"{by_rule[n].agree:.3f}" for n in names])
        false_positives.append([label] + [f"{by_rule_null[n].decided:.3f}" for n in names])
        dump.append({"pair": label, "real": real.to_dict(), "null": null.to_dict()})

    print(f"Orders per pair: {args.orders}, seed {args.seed}, alpha {args.alpha:g}.\n")
    print("Pairs (resolved items out of 500, gap in points, discordant items, exact McNemar p):\n")
    print(
        table(
            ["pair (A vs B)", "A", "B", "gap", "only A", "only B", "p on all 500"],
            summary,
        )
    )
    print("\nMean share of the 500 items used (share of orders that reached a decision):\n")
    print(table(["pair", *names], shares))
    print("\nShare of orders whose conclusion matches the exact McNemar test on all 500:\n")
    print(table(["pair", *names], agreement))
    print("\nFalse-positive rate with labels swapped at random within each pair:\n")
    print(table(["pair", *names], false_positives))
    if args.json is not None:
        args.json.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    if args.chart is not None:
        draw(dump, [row[3] for row in summary], args.alpha, args.chart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
