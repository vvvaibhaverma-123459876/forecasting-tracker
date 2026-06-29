"""Expected Calibration Error (ECE) computation."""

from __future__ import annotations

import math


def expected_calibration_error(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute ECE using equal-width probability bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: Predicted probabilities in [0, 1].
        outcomes: True binary outcomes aligned with confidences.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        ECE in [0, 1].  Lower is better.
    """
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")
    if not confidences:
        return 0.0

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, o in zip(confidences, outcomes):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"confidence {p} is outside [0, 1]")
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))

    n_total = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(p for p, _ in bucket) / len(bucket)
        avg_acc = sum(1 for _, o in bucket if o) / len(bucket)
        ece += (len(bucket) / n_total) * abs(avg_acc - avg_conf)

    return ece


def ece_decomposed(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> list[dict]:
    """Return per-bin calibration statistics for plotting.

    Returns a list of dicts with keys:
        bin_lower, bin_upper, avg_confidence, avg_accuracy, count
    """
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")

    bin_width = 1.0 / n_bins
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, o in zip(confidences, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))

    result = []
    for i, bucket in enumerate(bins):
        lower = i * bin_width
        upper = lower + bin_width
        if bucket:
            avg_conf = sum(p for p, _ in bucket) / len(bucket)
            avg_acc = sum(1 for _, o in bucket if o) / len(bucket)
        else:
            avg_conf = (lower + upper) / 2
            avg_acc = math.nan
        result.append(
            {
                "bin_lower": round(lower, 4),
                "bin_upper": round(upper, 4),
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": avg_acc if not math.isnan(avg_acc) else None,
                "count": len(bucket),
            }
        )
    return result
