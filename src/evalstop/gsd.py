"""Group-sequential boundaries by alpha spending.

A group-sequential design fixes a small number of looks in advance and spends
the error rate ``alpha`` over them. With information fractions
``t_1 < ... < t_K = 1`` and a spending function ``f`` with ``f(1) = alpha``,
the two-sided boundaries ``z_1, ..., z_K`` are chosen so that, under the null,

    P(|Z_1| < z_1, ..., |Z_{j-1}| < z_{j-1}, |Z_j| >= z_j) = f(t_j) - f(t_{j-1}),

where ``(Z_j)`` has the joint distribution of a Brownian motion observed at the
times ``t_j`` and standardised. This is the alpha-spending approach of Lan and
DeMets. Two spending functions are provided:

* ``"obf"``: ``f(t) = 4 - 4 * Phi(z_{alpha/4} / sqrt(t))``, the one-sided
  function ``2 - 2 * Phi(z_{a/2} / sqrt(t))`` applied with ``a = alpha / 2`` to
  each side. It resembles the O'Brien-Fleming boundary (very strict early,
  close to the fixed-sample critical value at the end);
* ``"pocock"``: ``f(t) = alpha * log(1 + (e - 1) * t)``, which resembles the
  Pocock boundary (about the same critical value at every look).

The probabilities are computed by recursive numerical integration over the
continuation region with Gauss-Legendre quadrature.

The boundaries assume a normally distributed test statistic with independent
increments, which holds only approximately for pass/fail data, and they are
valid only at the planned looks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evalstop.sequence import check_alpha
from evalstop.stats import norm_cdf, norm_ppf

SPENDING = ("obf", "pocock")
_NODES = 320


def spending(kind: str, t: float, alpha: float) -> float:
    """Cumulative error rate spent by information fraction ``t`` in (0, 1]."""
    if kind == "obf":
        return 4.0 - 4.0 * float(norm_cdf(norm_ppf(1.0 - alpha / 4.0) / math.sqrt(t)))
    if kind == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    raise ValueError(f"unknown spending function {kind!r}; expected one of {', '.join(SPENDING)}")


@dataclass(frozen=True)
class GroupSequentialDesign:
    """Two-sided boundaries of a group-sequential design.

    ``z[j]`` is the critical value for ``|Z|`` at look ``j`` and ``spent[j]``
    the cumulative error rate used up to and including that look.
    """

    kind: str
    alpha: float
    fractions: tuple[float, ...]
    z: tuple[float, ...]
    spent: tuple[float, ...]

    @property
    def nominal_p(self) -> tuple[float, ...]:
        """Two-sided p-value a single look needs to cross its boundary."""
        return tuple(2.0 * (1.0 - float(norm_cdf(value))) for value in self.z)


def _pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def design(
    looks: int | Sequence[float], alpha: float = 0.05, kind: str = "obf"
) -> GroupSequentialDesign:
    """Boundaries for equally spaced looks (an integer) or given information fractions."""
    alpha = check_alpha(alpha)
    if isinstance(looks, (int, np.integer)):
        if looks < 1:
            raise ValueError("looks must be at least 1")
        fractions = [(j + 1) / looks for j in range(int(looks))]
    else:
        fractions = [float(t) for t in looks]
        increasing = all(a < b for a, b in zip(fractions, fractions[1:], strict=False))
        if not fractions or fractions[0] <= 0.0 or not increasing or fractions[-1] > 1.0 + 1e-12:
            raise ValueError("information fractions must increase within (0, 1]")
    spent = [spending(kind, t, alpha) for t in fractions]
    nodes, weights = np.polynomial.legendre.leggauss(_NODES)

    z_first = norm_ppf(1.0 - spent[0] / 2.0)
    bound = z_first * math.sqrt(fractions[0])
    z = [z_first]
    x = bound * nodes
    w = bound * weights
    density = _pdf(x / math.sqrt(fractions[0])) / math.sqrt(fractions[0])
    for j in range(1, len(fractions)):
        step = math.sqrt(fractions[j] - fractions[j - 1])
        target = spent[j] - spent[j - 1]
        mass = w * density

        def exit_probability(b: float, mass=mass, x=x, step=step) -> float:
            return 2.0 * float(np.sum(mass * (1.0 - norm_cdf((b - x) / step))))

        low, high = 0.0, 12.0 * math.sqrt(fractions[j])
        if target <= exit_probability(high):
            bound = high
        else:
            for _ in range(80):
                mid = 0.5 * (low + high)
                if exit_probability(mid) > target:
                    low = mid
                else:
                    high = mid
            bound = 0.5 * (low + high)
        z.append(bound / math.sqrt(fractions[j]))
        if j + 1 < len(fractions):
            new_x = bound * nodes
            kernel = _pdf((new_x[:, None] - x[None, :]) / step) / step
            density = kernel @ mass
            x = new_x
            w = bound * weights
    return GroupSequentialDesign(kind, alpha, tuple(fractions), tuple(z), tuple(spent))


@dataclass(frozen=True)
class DesignReport:
    """Everything ``evalstop design`` prints, as data."""

    plan: GroupSequentialDesign
    items: int | None = None

    def _item_counts(self) -> list[int | None]:
        if self.items is None:
            return [None] * len(self.plan.z)
        return [math.ceil(self.items * t - 1e-9) for t in self.plan.fractions]

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary."""
        plan = self.plan
        return {
            "command": "design",
            "spending": plan.kind,
            "alpha": plan.alpha,
            "items": self.items,
            "looks": [
                {
                    "look": j + 1,
                    "information_fraction": round(plan.fractions[j], 6),
                    "items": count,
                    "z": round(plan.z[j], 4),
                    "nominal_p": round(plan.nominal_p[j], 6),
                    "alpha_spent": round(plan.spent[j], 6),
                }
                for j, count in enumerate(self._item_counts())
            ],
        }

    def to_markdown(self) -> str:
        """The report as Markdown."""
        plan = self.plan
        name = "O'Brien-Fleming type" if plan.kind == "obf" else "Pocock type"
        lines = [
            "# evalstop design",
            "",
            f"- {len(plan.z)} equally spaced looks, alpha {plan.alpha:g}, two-sided, "
            f"{name} spending",
            "",
            "| look | share of items | items | reject when abs(z) >= | nominal p | alpha spent |",
            "|---|---|---|---|---|---|",
        ]
        for j, count in enumerate(self._item_counts()):
            lines.append(
                f"| {j + 1} | {plan.fractions[j]:.3f} | {'-' if count is None else count} "
                f"| {plan.z[j]:.4f} | {plan.nominal_p[j]:.5f} | {plan.spent[j]:.5f} |"
            )
        lines += [
            "",
            "z is the test statistic at that look, for example (wins - losses) / sqrt(wins + "
            "losses) over the discordant pairs. The error rate holds only if the data are "
            "examined at these looks and nowhere else.",
        ]
        return "\n".join(lines) + "\n"
