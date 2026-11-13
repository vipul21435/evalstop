"""Small helpers for report text."""

from __future__ import annotations


def count(n: int, noun: str) -> str:
    """``n`` followed by ``noun``, singular when ``n`` is 1: ``1 score``, ``2 scores``."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
