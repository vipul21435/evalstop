"""Sequential monitoring for evaluations.

Confidence sequences that stay valid however often a running evaluation is
inspected, so a run can stop once the answer is settled.
"""

from evalstop.betting import BettingCS, betting_cs
from evalstop.mixture import MixtureCS, best_rho, mixture_cs, normal_mixture_boundary
from evalstop.sequence import ConfidenceSequence, OnlineCS, Stop

__version__ = "0.1.0"

__all__ = [
    "BettingCS",
    "ConfidenceSequence",
    "MixtureCS",
    "OnlineCS",
    "Stop",
    "__version__",
    "best_rho",
    "betting_cs",
    "mixture_cs",
    "normal_mixture_boundary",
]
