"""Download per-instance results of public SWE-bench Verified submissions.

Reads the public SWE-bench/experiments repository on GitHub and writes one
JSON Lines file with the fields item, model, score that evalstop reads.

That repository has no licence file, so nothing from it is stored in this
one: only this script and numbers derived from its output.

    python scripts/fetch_swebench_verified.py --out verified.jsonl

Standard library only. Two layouts are understood: a submission folder with
``per_instance_details.json`` (every instance with a ``resolved`` flag) and
the older ``results/results.json`` (a list of resolved instance ids).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"

# Submissions that share one agent scaffold, so pairs differ only in the model.
DEFAULT_SUBMISSIONS = (
    "20260217_mini-v2.0.0_deepseek-3-2-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high",
    "20260217_mini-v2.0.0_glm-5-high",
    "20260217_mini-v2.0.0_gpt-5-2-high",
    "20260217_mini-v2.0.0_gpt-5-mini",
    "20260217_mini-v2.0.0_kimi-k2-5-high",
    "20260217_mini-v2.0.0_minimax-2-5-high",
)


def fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "evalstop-fetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def resolved_ids(name: str) -> tuple[set[str], set[str]]:
    """Return (resolved instance ids, every instance id the submission lists)."""
    try:
        details = fetch_json(f"{BASE}/{name}/per_instance_details.json")
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
        try:
            results = fetch_json(f"{BASE}/{name}/results/results.json")
        except urllib.error.HTTPError as inner:
            if inner.code != 404:
                raise
            raise SystemExit(f"{name}: no per-instance results in a known layout") from None
        resolved = set(results["resolved"])
        listed = set().union(*(set(ids) for ids in results.values() if isinstance(ids, list)))
        return resolved, listed
    resolved = {item for item, entry in details.items() if entry.get("resolved")}
    return resolved, set(details)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON Lines file to write")
    parser.add_argument(
        "--submissions",
        nargs="+",
        default=list(DEFAULT_SUBMISSIONS),
        help="folder names under evaluation/verified (default: seven with one scaffold)",
    )
    args = parser.parse_args(argv)

    resolved: dict[str, set[str]] = {}
    items: set[str] = set()
    for name in args.submissions:
        resolved[name], listed = resolved_ids(name)
        items |= listed
        print(f"{name}: {len(resolved[name])} resolved, {len(listed)} listed", file=sys.stderr)
    ordered = sorted(items)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in ordered:
            for name in args.submissions:
                row = {"item": item, "model": name, "score": int(item in resolved[name])}
                handle.write(json.dumps(row) + "\n")
    print(f"wrote {args.out.name}: {len(ordered)} items, {len(args.submissions)} submissions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
