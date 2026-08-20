"""Forecast-vs-actuals degradation monitor.

As new actuals arrive, compute rolling MASE of the stored forecasts against
them; alert when it crosses the configured threshold (e.g. demand regime
change, stockouts, data issues).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from demandcast.evaluation.metrics import summarize
from demandcast.settings import get_config


@dataclass
class AccuracyStatus:
    mase: float
    threshold: float
    degraded: bool
    n_points: int


def check_accuracy(
    actuals: pd.DataFrame, forecasts: pd.DataFrame, train: pd.DataFrame
) -> AccuracyStatus:
    """Compare realized demand vs stored forecasts over their overlap."""
    threshold = get_config()["monitoring"]["degradation_threshold"]
    quantiles = get_config()["forecast"]["quantiles"]
    metrics = summarize(actuals, forecasts, train, quantiles)
    overlap = actuals.merge(forecasts, on=["unique_id", "ds"], how="inner")
    return AccuracyStatus(
        mase=metrics["mase"],
        threshold=threshold,
        degraded=metrics["mase"] > threshold,
        n_points=len(overlap),
    )
