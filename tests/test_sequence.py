"""Stopping rules on a confidence sequence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalstop import ConfidenceSequence, MixtureCS, Stop


def make(lower, upper, bounds=(0.0, 1.0)) -> ConfidenceSequence:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return ConfidenceSequence(
        lower=lower,
        upper=upper,
        mean=(lower + upper) / 2.0,
        alpha=0.05,
        method="test",
        bounds=bounds,
    )


def test_threshold_stop_reports_the_first_look_and_the_direction():
    seq = make([0.1, 0.3, 0.55, 0.6], [0.9, 0.8, 0.75, 0.7])
    assert seq.threshold_stop(0.5) == Stop(n=3, decision="above", lower=0.55, upper=0.75)
    assert seq.threshold_stop(0.78) == Stop(n=3, decision="below", lower=0.55, upper=0.75)
    assert seq.threshold_stop(0.65) is None


def test_a_threshold_equal_to_a_limit_counts_as_excluded():
    # The limits are the closure of an open accepted set, so a value equal to a
    # limit has been rejected.
    seq = make([0.2, 0.5], [0.9, 0.8])
    assert seq.threshold_stop(0.5) == Stop(n=2, decision="above", lower=0.5, upper=0.8)
    assert seq.threshold_stop(0.9) == Stop(n=1, decision="below", lower=0.2, upper=0.9)


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, 1.2])
def test_threshold_must_be_strictly_inside_the_bounds(threshold):
    with pytest.raises(ValueError, match="threshold"):
        make([0.1], [0.9]).threshold_stop(threshold)


def test_width_stop_reports_the_first_look_that_is_narrow_enough():
    seq = make([0.1, 0.3, 0.55, 0.6], [0.9, 0.8, 0.75, 0.7])
    assert seq.width_stop(0.2) == Stop(n=3, decision="width", lower=0.55, upper=0.75)
    assert seq.width_stop(0.05) is None
    with pytest.raises(ValueError, match="width"):
        seq.width_stop(0.0)


def test_an_empty_interval_is_reported_and_never_counts_as_narrow():
    seq = make([0.1, 0.5, 0.7], [0.9, 0.6, 0.6])
    assert seq.first_empty == 3
    np.testing.assert_allclose(seq.width, [0.8, 0.1, 0.0])
    assert seq.width_stop(0.05) is None
    assert make([0.1], [0.9]).first_empty is None


def test_online_state_before_and_after_the_first_score():
    cs = MixtureCS(0.05, lo=-1.0, hi=1.0)
    assert cs.n == 0
    assert math.isnan(cs.mean)
    assert cs.interval == (-1.0, 1.0)
    cs.update(1.0)
    assert cs.n == 1
    assert cs.mean == 1.0
    assert cs.lower <= cs.upper == 1.0
