"""Command line interface: ``evalstop monitor``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from evalstop import __version__
from evalstop.io import FORMATS, read_records, scores_for
from evalstop.monitor import METHODS, monitor_scores


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalstop",
        description="Monitor a running evaluation with intervals that stay valid at every look.",
    )
    parser.add_argument("--version", action="version", version=f"evalstop {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    monitor = commands.add_parser(
        "monitor",
        help="confidence sequence for one model's mean score, with stopping rules",
        description=(
            "Read per-item results in evaluation order and report a confidence sequence "
            "for the mean score, plus the look at which each stopping rule fired."
        ),
    )
    monitor.add_argument(
        "results",
        help="JSON Lines or CSV with the fields item, model, score; '-' reads standard input",
    )
    monitor.add_argument("--model", help="model to monitor when the results hold several")
    monitor.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="error rate over all looks together (default: 0.05)",
    )
    monitor.add_argument(
        "--method",
        choices=METHODS,
        default="betting",
        help="betting adapts to the variance; mixture is a closed-form normal-mixture "
        "boundary (default: betting)",
    )
    monitor.add_argument(
        "--threshold",
        type=float,
        help="report when the mean is first known to be above or below this value",
    )
    monitor.add_argument(
        "--width",
        type=float,
        help="report when the interval is first no wider than this",
    )
    monitor.add_argument("--lo", type=float, default=0.0, help="smallest score (default: 0)")
    monitor.add_argument("--hi", type=float, default=1.0, help="largest score (default: 1)")
    monitor.add_argument(
        "--grid",
        type=int,
        default=1000,
        help="betting: number of candidate means between lo and hi (default: 1000)",
    )
    monitor.add_argument(
        "--tune-at",
        type=float,
        default=200,
        help="mixture: number of scores at which the interval is tightest (default: 200)",
    )
    monitor.add_argument(
        "--every",
        type=int,
        help="print a trace row every this many scores (default: about ten rows)",
    )
    monitor.add_argument(
        "--input-format",
        choices=FORMATS,
        help="format of the results (default: from the file suffix or the first line)",
    )
    monitor.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output format (default: markdown)",
    )
    monitor.set_defaults(run=_monitor)
    return parser


def _monitor(args: argparse.Namespace) -> str:
    records = read_records(args.results, args.input_format)
    model, scores = scores_for(records, args.model)
    report = monitor_scores(
        scores,
        alpha=args.alpha,
        method=args.method,
        threshold=args.threshold,
        width=args.width,
        lo=args.lo,
        hi=args.hi,
        grid=args.grid,
        tune_at=args.tune_at,
        every=args.every,
        source="standard input" if args.results == "-" else args.results,
        model=model,
    )
    if args.format == "json":
        return json.dumps(report.to_dict(), indent=2) + "\n"
    return report.to_markdown()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit status."""
    args = _build_parser().parse_args(argv)
    try:
        output = args.run(args)
    except (OSError, ValueError) as err:
        print(f"evalstop: error: {err}", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
