"""Brier score computation."""

from __future__ import annotations


def brier_score(confidence: float, outcome: bool) -> float:
    """Compute the Brier score for a single prediction.

    Brier score = (p - o)^2  where p is probability and o is 0/1 outcome.
    Lower is better; perfect calibration -> 0.0, random -> 0.25.

    Args:
        confidence: Predicted probability in [0, 1].
        outcome: Actual binary outcome.

    Returns:
        Brier score in [0, 1].
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    o = 1.0 if outcome else 0.0
    return (confidence - o) ** 2


def rolling_brier_average(scores: list[float]) -> float:
    """Return the mean Brier score over a list of scores."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
