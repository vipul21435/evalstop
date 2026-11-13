"""Command line interface: ``evalstop monitor | compare | simulate | replay``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from evalstop import __version__, rules
from evalstop.io import FORMATS, read_records, scores_for
from evalstop.monitor import METHODS, monitor_scores
from evalstop.paired import compare_pairs, pair_records
from evalstop.simulate import replay, simulate


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "results",
        help="JSON Lines or CSV with the fields item, model, score; '-' reads standard input",
    )
    parser.add_argument(
        "--input-format",
        choices=FORMATS,
        help="format of the results (default: from the file suffix or the first line)",
    )


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output format (default: markdown)",
    )


def _add_alpha(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="error rate over all looks together (default: 0.05)",
    )


def _add_sequence(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="betting",
        help="betting adapts to the variance; mixture is a closed-form normal-mixture "
        "boundary (default: betting)",
    )
    parser.add_argument("--lo", type=float, default=0.0, help="smallest score (default: 0)")
    parser.add_argument("--hi", type=float, default=1.0, help="largest score (default: 1)")
    parser.add_argument(
        "--grid",
        type=int,
        default=1000,
        help="betting: number of candidate values on the grid (default: 1000)",
    )
    parser.add_argument(
        "--every",
        type=int,
        help="print a trace row every this many scores (default: about ten rows)",
    )


def _add_tuning(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tune-at",
        type=float,
        default=200,
        help="mixture: number of scores at which the interval is tightest (default: 200)",
    )


def _add_pair(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-a", help="first model (default: the first of two in the file)")
    parser.add_argument("--model-b", help="second model (default: the second of two in the file)")
    parser.add_argument(
        "--min-effect",
        type=float,
        help="stop for futility once every difference of at least this size is excluded",
    )


def _add_rules(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=rules.RULES,
        default=list(rules.RULES),
        metavar="RULE",
        help=f"rules to run (default: all of {', '.join(rules.RULES)})",
    )
    parser.add_argument(
        "--look-every",
        type=int,
        default=1,
        help="naive rule: number of items between looks (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")


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
    _add_input(monitor)
    monitor.add_argument("--model", help="model to monitor when the results hold several")
    _add_alpha(monitor)
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
    _add_sequence(monitor)
    _add_tuning(monitor)
    _add_output(monitor)
    monitor.set_defaults(run=_monitor)

    compare = commands.add_parser(
        "compare",
        help="two models on the same items: sequence on the difference and sequential tests",
        description=(
            "Pair two models by item and report a confidence sequence for the mean "
            "difference, a sequential test on discordant pairs, a futility rule, and for "
            "contrast the exact McNemar test at every look and on all pairs."
        ),
    )
    _add_input(compare)
    _add_pair(compare)
    _add_alpha(compare)
    _add_sequence(compare)
    _add_tuning(compare)
    _add_output(compare)
    compare.set_defaults(run=_compare)

    sim = commands.add_parser(
        "simulate",
        help="false-positive rate, power and items used by each rule, by seeded simulation",
        description=(
            "Simulate paired pass/fail results at the given effect sizes and report how "
            "often each rule decides and how many items it uses."
        ),
    )
    sim.add_argument(
        "--effects",
        type=float,
        nargs="+",
        default=[0.0, 0.1],
        help="differences in pass rate to simulate; 0 gives the false-positive rate "
        "(default: 0 0.1)",
    )
    sim.add_argument("--items", type=int, default=500, help="items per run (default: 500)")
    sim.add_argument("--runs", type=int, default=2000, help="runs per effect (default: 2000)")
    sim.add_argument(
        "--discordance",
        type=float,
        default=0.3,
        help="share of items on which the two models disagree (default: 0.3)",
    )
    sim.add_argument("--min-effect", type=float, help="futility margin for the sequences")
    _add_alpha(sim)
    _add_tuning(sim)
    _add_rules(sim)
    _add_output(sim)
    sim.set_defaults(run=_simulate)

    rep = commands.add_parser(
        "replay",
        help="run the rules over real paired results in random orders",
        description=(
            "Shuffle the items many times and report how many items each rule needed and "
            "whether it agreed with the exact McNemar test on all pairs. With --null the "
            "labels are swapped at random within each pair, so every decision is a false "
            "positive."
        ),
    )
    _add_input(rep)
    _add_pair(rep)
    rep.add_argument("--orders", type=int, default=1000, help="random orders (default: 1000)")
    rep.add_argument("--null", action="store_true", help="swap labels at random within each pair")
    _add_alpha(rep)
    _add_tuning(rep)
    _add_rules(rep)
    _add_output(rep)
    rep.set_defaults(run=_replay)
    return parser


def _source(args: argparse.Namespace) -> str:
    return "standard input" if args.results == "-" else args.results


def _monitor(args: argparse.Namespace):
    records = read_records(args.results, args.input_format)
    model, scores = scores_for(records, args.model)
    return monitor_scores(
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
        source=_source(args),
        model=model,
    )


def _compare(args: argparse.Namespace):
    records = read_records(args.results, args.input_format)
    pairs = pair_records(records, args.model_a, args.model_b)
    return compare_pairs(
        pairs,
        alpha=args.alpha,
        method=args.method,
        min_effect=args.min_effect,
        lo=args.lo,
        hi=args.hi,
        grid=args.grid,
        tune_at=args.tune_at,
        every=args.every,
        source=_source(args),
    )


def _simulate(args: argparse.Namespace):
    return simulate(
        effects=args.effects,
        items=args.items,
        runs=args.runs,
        discordance=args.discordance,
        alpha=args.alpha,
        rule_names=args.rules,
        min_effect=args.min_effect,
        tune_at=args.tune_at,
        every=args.look_every,
        seed=args.seed,
    )


def _replay(args: argparse.Namespace):
    records = read_records(args.results, args.input_format)
    pairs = pair_records(records, args.model_a, args.model_b)
    return replay(
        pairs,
        orders=args.orders,
        alpha=args.alpha,
        rule_names=args.rules,
        null=args.null,
        min_effect=args.min_effect,
        tune_at=args.tune_at,
        every=args.look_every,
        seed=args.seed,
        source=_source(args),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit status."""
    args = _build_parser().parse_args(argv)
    try:
        report = args.run(args)
        if args.format == "json":
            output = json.dumps(report.to_dict(), indent=2) + "\n"
        else:
            output = report.to_markdown()
    except (OSError, ValueError) as err:
        print(f"evalstop: error: {err}", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
