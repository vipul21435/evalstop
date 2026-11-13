"""Operating characteristics of the stopping rules: seeded simulation and replay.

``simulate`` draws paired pass/fail results from a model with a chosen effect
and reports, for each rule, how often it decides (false positives when the
effect is zero, power otherwise) and how many items it uses.

``replay`` runs the rules over real paired results in random orders and
reports how many items each rule needed and whether its decision agreed with
the fixed-sample test on all the data. With ``null=True`` the two labels are
swapped at random within each pair first, which makes "no difference" true by
construction, so every decision is a false positive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evalstop import rules
from evalstop.paired import Pairs
from evalstop.sequence import check_alpha
from evalstop.stats import mcnemar_exact_p

_VALID = {
    "betting": "yes",
    "mixture": "yes",
    "discordant": "yes",
    "obf": "planned looks",
    "pocock": "planned looks",
    # Wald's thresholds bound the false-positive rate by alpha / (1 - beta), not alpha.
    "sprt": "yes, at alpha/(1-beta)",
    "naive": "no",
    "fixed": "one look",
}
_BINARY_ONLY = ("discordant", "obf", "pocock", "sprt", "naive", "fixed")
# Largest number of difference values held in memory at once.
_CHUNK_CELLS = 4_000_000


def _check_rules(names: Sequence[str]) -> tuple[str, ...]:
    for name in names:
        if name not in rules.RULES:
            raise ValueError(f"unknown rule {name!r}; expected one of {', '.join(rules.RULES)}")
    if not names:
        raise ValueError("choose at least one rule")
    return tuple(names)


def _run_all(
    names: Sequence[str], d: np.ndarray, alpha: float, options: dict[str, object]
) -> dict[str, rules.Outcome]:
    return {name: rules.run_rule(name, d, alpha, **options) for name in names}


def _options(
    min_effect: float | None, tune_at: float, every: int, looks: int, beta: float, win_rate: float
) -> dict[str, object]:
    return {
        "min_effect": min_effect,
        "tune_at": tune_at,
        "every": every,
        "looks": looks,
        "beta": beta,
        "win_rate": win_rate,
    }


def _merge(parts: list[dict[str, rules.Outcome]]) -> dict[str, rules.Outcome]:
    return {
        name: rules.Outcome(
            np.concatenate([part[name].stop for part in parts]),
            np.concatenate([part[name].decision for part in parts]),
        )
        for name in parts[0]
    }


@dataclass(frozen=True)
class SimulationRow:
    """One rule at one effect size.

    ``right`` and ``wrong`` are the shares of runs that decided for the truly
    better model and against it. At effect 0 every decision is wrong, so
    ``wrong`` is the false-positive rate there.
    """

    effect: float
    rule: str
    right: float
    wrong: float
    futile: float
    mean_items: float


@dataclass(frozen=True)
class SimulationReport:
    """Everything ``evalstop simulate`` prints, as data."""

    items: int
    runs: int
    alpha: float
    discordance: float
    min_effect: float | None
    every: int
    seed: int
    rows: tuple[SimulationRow, ...]

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary."""
        return {
            "command": "simulate",
            "items": self.items,
            "runs": self.runs,
            "alpha": self.alpha,
            "discordance": self.discordance,
            "min_effect": self.min_effect,
            "every": self.every,
            "seed": self.seed,
            "max_standard_error": round(0.5 / math.sqrt(self.runs), 6),
            "rows": [
                {
                    "effect": row.effect,
                    "rule": row.rule,
                    "valid_at_every_look": _VALID[row.rule],
                    "false_positive_rate": round(row.wrong, 6),
                    "power": None if row.effect == 0 else round(row.right, 6),
                    "futility_stops": round(row.futile, 6),
                    "mean_items": round(row.mean_items, 3),
                }
                for row in self.rows
            ],
        }

    def to_markdown(self) -> str:
        """The report as Markdown."""
        futility = ""
        if self.min_effect is not None:
            futility = f", futility below a difference of {self.min_effect:g}"
        looks = "" if self.every == 1 else f", naive looks every {self.every} items"
        error = 0.5 / math.sqrt(self.runs)
        lines = [
            "# evalstop simulate",
            "",
            f"- {self.runs} runs of {self.items} paired pass/fail items per effect, "
            f"seed {self.seed}",
            f"- the two models disagree on a share {self.discordance:g} of the items; "
            "the effect is the difference between their pass rates",
            f"- alpha {self.alpha:g}{futility}{looks}",
            "",
            "| effect | rule | valid at every look | false positives | power | futility stops "
            "| mean items |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            power = "-" if row.effect == 0 else f"{row.right:.3f}"
            lines.append(
                f"| {row.effect:g} | {row.rule} | {_VALID[row.rule]} | {row.wrong:.3f} | {power} "
                f"| {row.futile:.3f} | {row.mean_items:.1f} |"
            )
        lines += [
            "",
            "False positives are decisions for the wrong model (any decision when the effect "
            f"is 0). Monte Carlo standard error of each share: at most {error:.3f}.",
        ]
        return "\n".join(lines) + "\n"


def simulate(
    *,
    effects: Sequence[float] = (0.0, 0.1),
    items: int = 500,
    runs: int = 2000,
    discordance: float = 0.3,
    alpha: float = 0.05,
    rule_names: Sequence[str] = rules.RULES,
    min_effect: float | None = None,
    tune_at: float = 200,
    every: int = 1,
    looks: int = 5,
    beta: float = 0.2,
    win_rate: float = 0.65,
    seed: int = 0,
) -> SimulationReport:
    """Simulate paired pass/fail results and run every rule on them.

    Each pair is discordant with probability ``discordance``. Given an effect
    ``e`` (pass rate of A minus pass rate of B), A alone passes with
    probability ``(discordance + e) / 2`` and B alone with
    ``(discordance - e) / 2``. Only the differences matter to the rules, so
    the pass rates themselves are not needed.
    """
    alpha = check_alpha(alpha)
    names = _check_rules(rule_names)
    if items < 1 or runs < 1:
        raise ValueError("items and runs must be at least 1")
    if not 0.0 < discordance <= 1.0:
        raise ValueError("discordance must lie in (0, 1]")
    options = _options(min_effect, tune_at, every, looks, beta, win_rate)
    rng = np.random.default_rng(seed)
    chunk = max(1, min(runs, _CHUNK_CELLS // items))
    rows: list[SimulationRow] = []
    for effect in effects:
        if abs(effect) > discordance:
            raise ValueError(f"effect {effect:g} is larger than the discordance {discordance:g}")
        p_plus = (discordance + effect) / 2.0
        parts = []
        for start in range(0, runs, chunk):
            u = rng.random((min(chunk, runs - start), items))
            d = (u < p_plus).astype(float) - ((u >= p_plus) & (u < discordance)).astype(float)
            parts.append(_run_all(names, d, alpha, options))
        truth = int(np.sign(effect))
        for name, outcome in _merge(parts).items():
            decided = (outcome.decision == rules.A_BETTER) | (outcome.decision == rules.B_BETTER)
            right = decided & (outcome.decision == truth) if truth else np.zeros_like(decided)
            rows.append(
                SimulationRow(
                    effect=float(effect),
                    rule=name,
                    right=float(np.mean(right)),
                    wrong=float(np.mean(decided & ~right)),
                    futile=float(np.mean(outcome.decision == rules.FUTILE)),
                    mean_items=float(np.mean(outcome.stop)),
                )
            )
    return SimulationReport(
        items=items,
        runs=runs,
        alpha=alpha,
        discordance=discordance,
        min_effect=min_effect,
        every=every,
        seed=seed,
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class ReplayRow:
    """One rule over all replayed orders.

    ``decided`` is the share of orders in which the rule picked a model,
    ``agree`` the share in which its conclusion matched the fixed-sample test
    on all pairs (a futility stop counts as "no difference").
    """

    rule: str
    decided: float
    agree: float
    futile: float
    mean_share: float
    median_items: float


@dataclass(frozen=True)
class ReplayReport:
    """Everything ``evalstop replay`` prints, as data."""

    source: str
    model_a: str
    model_b: str
    pairs: int
    wins: int
    losses: int
    orders: int
    alpha: float
    null: bool
    min_effect: float | None
    every: int
    seed: int
    full_p: float
    full_decision: str | None
    rows: tuple[ReplayRow, ...]
    skipped: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary."""
        return {
            "command": "replay",
            "source": self.source,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "pairs": self.pairs,
            "a_only": self.wins,
            "b_only": self.losses,
            "orders": self.orders,
            "alpha": self.alpha,
            "null": self.null,
            "min_effect": self.min_effect,
            "every": self.every,
            "seed": self.seed,
            "full_data_p": round(self.full_p, 6),
            "full_data_decision": self.full_decision,
            "skipped_rules": list(self.skipped),
            "rows": [
                {
                    "rule": row.rule,
                    "valid_at_every_look": _VALID[row.rule],
                    "decided": round(row.decided, 6),
                    "agrees_with_full_data": None if self.null else round(row.agree, 6),
                    "futility_stops": round(row.futile, 6),
                    "mean_share_of_items": round(row.mean_share, 6),
                    "median_items": row.median_items,
                }
                for row in self.rows
            ],
        }

    def to_markdown(self) -> str:
        """The report as Markdown."""
        verdict = {"a": f"`{self.model_a}` better", "b": f"`{self.model_b}` better"}.get(
            self.full_decision or "", "no significant difference"
        )
        lines = [
            "# evalstop replay",
            "",
            f"- input: `{self.source}`, A `{self.model_a}`, B `{self.model_b}`",
            f"- pairs: {self.pairs}, only A passes {self.wins}, only B passes {self.losses}",
            f"- exact McNemar test on all pairs: p = {self.full_p:.3g} ({verdict} at alpha "
            f"{self.alpha:g})",
            f"- {self.orders} random orders, seed {self.seed}"
            + (", labels swapped at random within each pair (null)" if self.null else ""),
        ]
        if self.min_effect is not None:
            lines.append(f"- futility below a difference of {self.min_effect:g}")
        if self.skipped:
            lines.append(f"- skipped (scores are not pass/fail): {', '.join(self.skipped)}")
        lines.append("")
        if self.null:
            lines += [
                "| rule | valid at every look | false-positive rate | mean share of items |",
                "|---|---|---|---|",
            ]
            for row in self.rows:
                lines.append(
                    f"| {row.rule} | {_VALID[row.rule]} | {row.decided:.3f} "
                    f"| {row.mean_share:.3f} |"
                )
        else:
            lines += [
                "| rule | valid at every look | decided | same as full-data test "
                "| mean share of items | median items |",
                "|---|---|---|---|---|---|",
            ]
            for row in self.rows:
                lines.append(
                    f"| {row.rule} | {_VALID[row.rule]} | {row.decided:.3f} | {row.agree:.3f} "
                    f"| {row.mean_share:.3f} | {row.median_items:g} |"
                )
        lines += [
            "",
            "Monte Carlo standard error of each share: at most "
            f"{0.5 / math.sqrt(self.orders):.3f}.",
        ]
        return "\n".join(lines) + "\n"


def replay(
    pairs: Pairs,
    *,
    orders: int = 1000,
    alpha: float = 0.05,
    rule_names: Sequence[str] = rules.RULES,
    null: bool = False,
    min_effect: float | None = None,
    tune_at: float = 200,
    every: int = 1,
    looks: int = 5,
    beta: float = 0.2,
    win_rate: float = 0.65,
    seed: int = 0,
    source: str = "",
) -> ReplayReport:
    """Run the rules over ``pairs`` in ``orders`` random orders.

    Scores must lie in [0, 1]. Rules built on McNemar's test need pass/fail
    scores and are skipped otherwise.
    """
    alpha = check_alpha(alpha)
    names = _check_rules(rule_names)
    n = len(pairs)
    if n == 0:
        raise ValueError("no item has a score from both models")
    if orders < 1:
        raise ValueError("orders must be at least 1")
    diff = pairs.diff
    if np.abs(diff).max() > 1.0:
        raise ValueError("replay needs scores in [0, 1]")
    skipped: tuple[str, ...] = ()
    if not pairs.binary:
        skipped = tuple(name for name in names if name in _BINARY_ONLY)
        names = tuple(name for name in names if name not in _BINARY_ONLY)
        if not names:
            raise ValueError("the chosen rules need pass/fail scores")
    wins, losses = int(np.sum(diff > 0)), int(np.sum(diff < 0))
    full = int(rules.fixed_rule(diff, alpha).decision[0])
    options = _options(min_effect, tune_at, every, looks, beta, win_rate)
    rng = np.random.default_rng(seed)
    chunk = max(1, min(orders, _CHUNK_CELLS // n))
    parts = []
    for start in range(0, orders, chunk):
        size = min(chunk, orders - start)
        d = np.tile(diff, (size, 1))
        if null:
            d *= rng.choice(np.array([-1.0, 1.0]), size=d.shape)
        d = rng.permuted(d, axis=1)
        parts.append(_run_all(names, d, alpha, options))
    rows = []
    for name, outcome in _merge(parts).items():
        decision = np.where(outcome.decision == rules.FUTILE, rules.UNDECIDED, outcome.decision)
        rows.append(
            ReplayRow(
                rule=name,
                decided=float(np.mean(decision != rules.UNDECIDED)),
                agree=float(np.mean(decision == full)),
                futile=float(np.mean(outcome.decision == rules.FUTILE)),
                mean_share=float(np.mean(outcome.stop)) / n,
                median_items=float(np.median(outcome.stop)),
            )
        )
    return ReplayReport(
        source=source,
        model_a=pairs.model_a,
        model_b=pairs.model_b,
        pairs=n,
        wins=wins,
        losses=losses,
        orders=orders,
        alpha=alpha,
        null=null,
        min_effect=min_effect,
        every=every,
        seed=seed,
        full_p=mcnemar_exact_p(wins, losses),
        full_decision={rules.A_BETTER: "a", rules.B_BETTER: "b"}.get(full),
        rows=tuple(rows),
        skipped=skipped,
    )
