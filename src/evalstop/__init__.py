"""Sequential monitoring for evaluations.

Confidence sequences and sequential tests that stay valid however often a
running evaluation is inspected, so a run can stop once the answer is settled.
"""

from evalstop.betting import BettingCS, betting_cs
from evalstop.gsd import GroupSequentialDesign, design
from evalstop.mixture import MixtureCS, best_rho, mixture_cs, normal_mixture_boundary
from evalstop.monitor import monitor_scores
from evalstop.paired import Pairs, compare_pairs, pair_records
from evalstop.sequence import ConfidenceSequence, OnlineCS, Stop
from evalstop.sprt import BernoulliSPRT, sprt_bernoulli
from evalstop.stats import mcnemar_exact_p

__version__ = "0.1.0"

__all__ = [
    "BernoulliSPRT",
    "BettingCS",
    "ConfidenceSequence",
    "GroupSequentialDesign",
    "MixtureCS",
    "OnlineCS",
    "Pairs",
    "Stop",
    "__version__",
    "best_rho",
    "betting_cs",
    "compare_pairs",
    "design",
    "mcnemar_exact_p",
    "mixture_cs",
    "monitor_scores",
    "normal_mixture_boundary",
    "pair_records",
    "sprt_bernoulli",
]
