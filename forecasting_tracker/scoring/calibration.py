"""Calibration curve data generation."""

from __future__ import annotations

from forecasting_tracker.scoring.ece import ece_decomposed


def calibration_curve_data(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> dict:
    """Return calibration curve data suitable for plotting.

    Returns a dict with:
        bins: list of bin statistics (from ece_decomposed)
        perfect_x: x-axis points for perfect calibration reference line
        perfect_y: y-axis points for perfect calibration reference line
    """
    bins = ece_decomposed(confidences, outcomes, n_bins=n_bins)
    xs = [b["avg_confidence"] for b in bins if b["count"] > 0]
    ys = [b["avg_accuracy"] for b in bins if b["count"] > 0 and b["avg_accuracy"] is not None]
    return {
        "bins": bins,
        "plotted_confidence": xs,
        "plotted_accuracy": ys,
        "perfect_x": [0.0, 1.0],
        "perfect_y": [0.0, 1.0],
    }
