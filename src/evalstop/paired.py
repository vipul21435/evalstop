"""Compare two models on the same items.

The comparison works on the paired differences (model A minus model B):

* a confidence sequence on the mean difference, from the betting or the
  normal-mixture method; the comparison is settled when it excludes 0, and it
  is futile when it lies inside ``(-min_effect, min_effect)``;
* for pass/fail scores, a mixture likelihood-ratio test on the discordant
  pairs, the sequential analogue of McNemar's test;
* for contrast, the exact McNemar test repeated at every look (not valid) and
  once on all pairs (valid, but only at the end).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evalstop import rules
from evalstop.betting import betting_cs
from evalstop.io import InputError, Record, model_names
from evalstop.mixture import mixture_cs
from evalstop.sequence import ConfidenceSequence, check_bounds
from evalstop.stats import discordant_log_evalue, mcnemar_exact_p

METHODS = ("betting", "mixture")
_DECISIONS = {
    rules.A_BETTER: "a",
    rules.B_BETTER: "b",
    rules.FUTILE: "futile",
    rules.UNDECIDED: None,
}


@dataclass(frozen=True, eq=False)
class Pairs:
    """Scores of two models on the items both were run on, in evaluation order."""

    model_a: str
    model_b: str
    items: tuple[str, ...]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    unpaired: int = 0

    def __len__(self) -> int:
        return len(self.items)

    @property
    def diff(self) -> NDArray[np.float64]:
        """Score of model A minus score of model B, per item."""
        return self.a - self.b

    @property
    def binary(self) -> bool:
        """Whether every score is 0 or 1."""
        both = np.concatenate([self.a, self.b])
        return bool(np.all((both == 0.0) | (both == 1.0)))


def pair_records(
    records: Iterable[Record], model_a: str | None = None, model_b: str | None = None
) -> Pairs:
    """Pair the records of two models by item.

    With no names given the records must hold exactly two models, taken in
    order of first appearance. A pair enters the evaluation order when its
    second score arrives. Items scored by only one of the two models are
    counted in ``unpaired`` and left out.
    """
    records = list(records)
    names = model_names(records)
    if model_a is None and model_b is None:
        if len(names) != 2:
            listed = ", ".join(repr(name) for name in names) or "none"
            raise InputError(f"need two models to compare; found {listed}")
        model_a, model_b = names
    elif model_a is None or model_b is None:
        raise InputError("name both models to compare, or neither")
    if model_a == model_b:
        raise InputError("the two models to compare must differ")
    for name in (model_a, model_b):
        if name not in names:
            listed = ", ".join(repr(other) for other in names) or "none"
            raise InputError(f"no results for model {name!r}; found {listed}")
    seen: dict[str, dict[str, float]] = {model_a: {}, model_b: {}}
    order: list[str] = []
    for record in records:
        if record.model not in seen:
            continue
        mine = seen[record.model]
        if record.item in mine:
            raise InputError(f"model {record.model!r} has two scores for item {record.item!r}")
        mine[record.item] = record.score
        other = seen[model_b if record.model == model_a else model_a]
        if record.item in other:
            order.append(record.item)
    singles = len(seen[model_a]) + len(seen[model_b]) - 2 * len(order)
    return Pairs(
        model_a=model_a,
        model_b=model_b,
        items=tuple(order),
        a=np.asarray([seen[model_a][item] for item in order], dtype=float),
        b=np.asarray([seen[model_b][item] for item in order], dtype=float),
        unpaired=singles,
    )


@dataclass(frozen=True)
class RuleLine:
    """One row of the comparison: when a rule stopped and what it concluded.

    ``decision`` is ``"a"``, ``"b"``, ``"futile"`` or ``None`` (still open).
    ``valid`` says whether the rule keeps its error rate when checked at every
    look.
    """

    name: str
    label: str
    valid: bool
    stopped_at: int | None
    decision: str | None
    note: str = ""


@dataclass(frozen=True)
class CompareReport:
    """Everything ``evalstop compare`` prints, as data."""

    source: str
    pairs: Pairs
    sequence: ConfidenceSequence
    settings: dict[str, float]
    lines: tuple[RuleLine, ...]
    min_effect: float | None
    every: int

    def _trace_rows(self) -> list[int]:
        n = len(self.sequence)
        rows = list(range(self.every, n + 1, self.every))
        if not rows or rows[-1] != n:
            rows.append(n)
        return rows

    def _counts(self) -> dict[str, int] | None:
        pairs = self.pairs
        if not pairs.binary:
            return None
        a, b = pairs.a == 1.0, pairs.b == 1.0
        return {
            "both": int(np.sum(a & b)),
            "a_only": int(np.sum(a & ~b)),
            "b_only": int(np.sum(~a & b)),
            "neither": int(np.sum(~a & ~b)),
        }

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary (numbers rounded to six decimals)."""
        seq, pairs = self.sequence, self.pairs

        def r(value: float) -> float:
            return round(float(value), 6)

        return {
            "command": "compare",
            "source": self.source,
            "model_a": pairs.model_a,
            "model_b": pairs.model_b,
            "pairs": len(pairs),
            "unpaired": pairs.unpaired,
            "mean_a": r(pairs.a.mean()),
            "mean_b": r(pairs.b.mean()),
            "difference": r(seq.mean[-1]),
            "counts": self._counts(),
            "method": seq.method,
            "alpha": seq.alpha,
            "settings": dict(self.settings),
            "min_effect": self.min_effect,
            "lower": r(seq.lower[-1]),
            "upper": r(seq.upper[-1]),
            "first_empty": seq.first_empty,
            "rules": [
                {
                    "rule": line.name,
                    "valid_at_every_look": line.valid,
                    "stopped_at": line.stopped_at,
                    "decision": line.decision,
                    "note": line.note,
                }
                for line in self.lines
            ],
            "trace": [
                {
                    "n": t,
                    "difference": r(seq.mean[t - 1]),
                    "lower": r(seq.lower[t - 1]),
                    "upper": r(seq.upper[t - 1]),
                }
                for t in self._trace_rows()
            ],
        }

    def _decision_text(self, decision: str | None) -> str:
        if decision == "a":
            return f"`{self.pairs.model_a}` better"
        if decision == "b":
            return f"`{self.pairs.model_b}` better"
        if decision == "futile":
            return f"no difference of {self.min_effect:g} or more"
        return "undecided"

    def to_markdown(self) -> str:
        """The report as Markdown."""
        seq, pairs = self.sequence, self.pairs
        n = len(pairs)
        level = f"{100 * (1 - seq.alpha):g}%"
        settings = "".join(f", {name} {value:g}" for name, value in self.settings.items())
        label = "betting (hedged capital)" if seq.method == "betting" else "normal mixture"
        lines = [
            "# evalstop compare",
            "",
            f"- input: `{self.source}`",
            f"- A: `{pairs.model_a}`, mean {pairs.a.mean():.4f}; "
            f"B: `{pairs.model_b}`, mean {pairs.b.mean():.4f}",
            f"- pairs: {n}, mean difference A - B: {seq.mean[-1]:+.4f}",
        ]
        if pairs.unpaired:
            lines.append(f"- left out: {pairs.unpaired} scores for items only one model was run on")
        counts = self._counts()
        if counts is not None:
            lines.append(
                f"- both pass {counts['both']}, only A {counts['a_only']}, "
                f"only B {counts['b_only']}, neither {counts['neither']}"
            )
        lines += [
            f"- method: {label}, alpha {seq.alpha:g}{settings}",
            f"- {level} confidence sequence for the difference after {n} pairs: "
            f"[{seq.lower[-1]:+.4f}, {seq.upper[-1]:+.4f}]",
            "",
            "## Rules",
            "",
            "| rule | valid at every look | stopped after | decision |",
            "|---|---|---|---|",
        ]
        for line in self.lines:
            when = f"{line.stopped_at} of {n}"
            if line.stopped_at is None:
                when = f"not within {n} pairs"
            note = f" ({line.note})" if line.note else ""
            lines.append(
                f"| {line.label} | {'yes' if line.valid else 'no'} | {when} "
                f"| {self._decision_text(line.decision)}{note} |"
            )
        lines += [
            "",
            "## Trace",
            "",
            "| pairs | difference | lower | upper |",
            "|---|---|---|---|",
        ]
        for t in self._trace_rows():
            i = t - 1
            lines.append(
                f"| {t} | {seq.mean[i]:+.4f} | {seq.lower[i]:+.4f} | {seq.upper[i]:+.4f} |"
            )
        if seq.first_empty is not None:
            lines += [
                "",
                f"Warning: after {seq.first_empty} pairs the running intersection became empty. "
                "The differences do not look like draws with one fixed mean, so the limits from "
                "that point on are not meaningful.",
            ]
        return "\n".join(lines) + "\n"


def _line(name: str, label: str, valid: bool, outcome: rules.Outcome, n: int, note: str = ""):
    decision = _DECISIONS[int(outcome.decision[0])]
    stopped = int(outcome.stop[0]) if decision is not None else None
    return RuleLine(name, label, valid, stopped, decision, note)


def compare_pairs(
    pairs: Pairs,
    *,
    alpha: float = 0.05,
    method: str = "betting",
    min_effect: float | None = None,
    lo: float = 0.0,
    hi: float = 1.0,
    grid: int = 1000,
    tune_at: float = 200,
    every: int | None = None,
    population: int | None = None,
    source: str = "",
) -> CompareReport:
    """Run every comparison rule over ``pairs`` in their evaluation order.

    ``lo`` and ``hi`` bound a single score, so a difference lies in
    ``[lo - hi, hi - lo]``. ``min_effect`` turns on the futility rule.
    ``population`` is the number of paired items in the whole benchmark; with
    it the betting sequence covers the mean difference over exactly those.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("no item has a score from both models")
    lo, hi = check_bounds(lo, hi)
    scale = hi - lo
    for scores in (pairs.a, pairs.b):
        if scores.min() < lo or scores.max() > hi:
            raise ValueError(f"scores must lie in [{lo:g}, {hi:g}]")
    if min_effect is not None and not 0.0 < min_effect < scale:
        raise ValueError(f"min_effect must lie strictly between 0 and {scale:g}")
    diff = pairs.diff
    if method == "betting":
        sequence = betting_cs(
            diff, alpha, lo=-scale, hi=scale, grid=grid, population=population
        )
        settings: dict[str, float] = {"grid": grid}
        label = "betting sequence on the difference"
        if population is not None:
            settings["population"] = population
            label = "betting sequence on the difference over the whole benchmark"
    elif method == "mixture":
        if population is not None:
            raise ValueError("population is only supported by the betting method")
        sequence = mixture_cs(diff, alpha, lo=-scale, hi=scale, tune_at=tune_at)
        settings = {"tune_at": tune_at}
        label = "mixture sequence on the difference"
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")

    settled = sequence.threshold_stop(0.0)
    futile = None if min_effect is None else sequence.inside_stop(-min_effect, min_effect)
    if settled is not None and (futile is None or settled.n <= futile.n):
        winner = "a" if settled.decision == "above" else "b"
        first = RuleLine(method, label, True, settled.n, winner)
    elif futile is not None:
        first = RuleLine(method, label, True, futile.n, "futile")
    else:
        first = RuleLine(method, label, True, None, None)
    lines = [first]

    if pairs.binary:
        wins, losses = int(np.sum(diff > 0)), int(np.sum(diff < 0))
        log_e = float(discordant_log_evalue(wins, losses))
        lines.append(
            _line(
                "discordant",
                "discordant pairs (sequential McNemar)",
                True,
                rules.discordant_rule(diff, alpha),
                n,
                f"final evidence ratio {math.exp(min(log_e, 700.0)):.3g}",
            )
        )
        lines.append(
            _line("naive", "exact McNemar at every look", False, rules.naive_rule(diff, alpha), n)
        )
        lines.append(
            _line(
                "fixed",
                "exact McNemar on all pairs",
                False,
                rules.fixed_rule(diff, alpha),
                n,
                f"p = {mcnemar_exact_p(wins, losses):.3g}, valid only as a single look",
            )
        )
    if every is None:
        every = max(1, math.ceil(n / 10))
    elif every < 1:
        raise ValueError("every must be at least 1")
    return CompareReport(
        source=source,
        pairs=pairs,
        sequence=sequence,
        settings=settings,
        lines=tuple(lines),
        min_effect=min_effect,
        every=int(every),
    )
