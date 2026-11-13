"""Monitor one model: a confidence sequence for its mean score, with stopping rules."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from evalstop.betting import betting_cs
from evalstop.mixture import mixture_cs
from evalstop.sequence import ConfidenceSequence, Stop

METHODS = ("betting", "mixture")
_LABELS = {"betting": "betting (hedged capital)", "mixture": "normal mixture"}


@dataclass(frozen=True)
class RuleResult:
    """Outcome of one stopping rule: ``stop`` is ``None`` when it never fired."""

    rule: str
    value: float
    stop: Stop | None


@dataclass(frozen=True)
class MonitorReport:
    """Everything ``evalstop monitor`` prints, as data."""

    source: str
    model: str
    sequence: ConfidenceSequence
    settings: dict[str, float]
    rules: tuple[RuleResult, ...]
    every: int

    def _trace_rows(self) -> list[int]:
        n = len(self.sequence)
        rows = list(range(self.every, n + 1, self.every))
        if not rows or rows[-1] != n:
            rows.append(n)
        return rows

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary (numbers rounded to six decimals)."""
        seq = self.sequence
        n = len(seq)

        def r(value: float) -> float:
            return round(float(value), 6)

        rules = []
        for result in self.rules:
            stop = result.stop
            rules.append(
                {
                    "rule": result.rule,
                    "value": result.value,
                    "stopped_at": None if stop is None else stop.n,
                    "decision": None if stop is None else stop.decision,
                    "lower": None if stop is None else r(stop.lower),
                    "upper": None if stop is None else r(stop.upper),
                }
            )
        trace = [
            {
                "n": t,
                "mean": r(seq.mean[t - 1]),
                "lower": r(seq.lower[t - 1]),
                "upper": r(seq.upper[t - 1]),
            }
            for t in self._trace_rows()
        ]
        return {
            "command": "monitor",
            "source": self.source,
            "model": self.model,
            "method": seq.method,
            "alpha": seq.alpha,
            "bounds": list(seq.bounds),
            "settings": dict(self.settings),
            "n": n,
            "mean": r(seq.mean[-1]),
            "lower": r(seq.lower[-1]),
            "upper": r(seq.upper[-1]),
            "first_empty": seq.first_empty,
            "rules": rules,
            "trace": trace,
        }

    def to_markdown(self) -> str:
        """The report as Markdown."""
        seq = self.sequence
        n = len(seq)
        lo, hi = seq.bounds
        level = f"{100 * (1 - seq.alpha):g}%"
        settings = "".join(f", {name} {value:g}" for name, value in self.settings.items())
        model = f", model `{self.model}`" if self.model else ""
        lines = [
            "# evalstop monitor",
            "",
            f"- input: `{self.source}`{model}",
            f"- scores: {n} in [{lo:g}, {hi:g}], mean {seq.mean[-1]:.4f}",
            f"- method: {_LABELS[seq.method]}, alpha {seq.alpha:g}{settings}",
            f"- {level} confidence sequence after {n} scores: "
            f"[{seq.lower[-1]:.4f}, {seq.upper[-1]:.4f}]",
        ]
        if self.rules:
            lines += [
                "",
                "## Stopping rules",
                "",
                "| rule | stopped after | decision | interval at that look |",
                "|---|---|---|---|",
            ]
            for result in self.rules:
                lines.append(_rule_row(result, n))
        lines += [
            "",
            "## Trace",
            "",
            "| scores | mean | lower | upper | width |",
            "|---|---|---|---|---|",
        ]
        for t in self._trace_rows():
            i = t - 1
            lines.append(
                f"| {t} | {seq.mean[i]:.4f} | {seq.lower[i]:.4f} | {seq.upper[i]:.4f} "
                f"| {seq.width[i]:.4f} |"
            )
        lines += [
            "",
            f"The interval holds at every look at once with probability at least {level}, "
            "provided the scores arrive in an order unrelated to their values.",
        ]
        if seq.first_empty is not None:
            lines += [
                "",
                f"Warning: after {seq.first_empty} scores the running intersection became empty. "
                "The scores do not look like draws with one fixed mean (for example the order "
                "is sorted by difficulty), so the limits from that point on are not meaningful.",
            ]
        return "\n".join(lines) + "\n"


def _rule_row(result: RuleResult, n: int) -> str:
    stop = result.stop
    if result.rule == "threshold":
        name = f"threshold {result.value:g}"
        decision = "undecided" if stop is None else f"mean {stop.decision} {result.value:g}"
    else:
        name = f"width <= {result.value:g}"
        decision = "not reached" if stop is None else "narrow enough"
    if stop is None:
        return f"| {name} | not within {n} scores | {decision} | |"
    interval = f"[{stop.lower:.4f}, {stop.upper:.4f}]"
    return f"| {name} | {stop.n} of {n} scores | {decision} | {interval} |"


def monitor_scores(
    scores: Iterable[float],
    *,
    alpha: float = 0.05,
    method: str = "betting",
    threshold: float | None = None,
    width: float | None = None,
    lo: float = 0.0,
    hi: float = 1.0,
    grid: int = 1000,
    tune_at: float = 200,
    every: int | None = None,
    source: str = "",
    model: str = "",
) -> MonitorReport:
    """Run one confidence sequence over ``scores`` and evaluate the stopping rules.

    ``threshold`` asks at which look the mean was first known to be above or
    below that value; ``width`` asks at which look the interval was first that
    narrow. ``every`` sets the spacing of the trace rows (about ten rows by
    default).
    """
    scores = list(scores)
    if not scores:
        raise ValueError("no scores to monitor")
    if method == "betting":
        sequence = betting_cs(scores, alpha, lo=lo, hi=hi, grid=grid)
        settings: dict[str, float] = {"grid": grid}
    elif method == "mixture":
        sequence = mixture_cs(scores, alpha, lo=lo, hi=hi, tune_at=tune_at)
        settings = {"tune_at": tune_at}
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")
    rules = []
    if threshold is not None:
        rules.append(RuleResult("threshold", threshold, sequence.threshold_stop(threshold)))
    if width is not None:
        rules.append(RuleResult("width", width, sequence.width_stop(width)))
    if every is None:
        every = max(1, math.ceil(len(sequence) / 10))
    elif every < 1:
        raise ValueError("every must be at least 1")
    return MonitorReport(
        source=source,
        model=model,
        sequence=sequence,
        settings=settings,
        rules=tuple(rules),
        every=int(every),
    )
