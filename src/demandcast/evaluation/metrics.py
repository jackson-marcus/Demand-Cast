"""Forecast metrics: MASE, RMSSE, pinball loss, interval coverage.

MASE/RMSSE scale errors by the training set's seasonal-naive error, so values
below 1.0 mean "better than repeating last week".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = 7


def _scale(train: pd.DataFrame) -> pd.Series:
    """Per-series mean |seasonal difference| on the training window."""

    def one(g: pd.Series) -> float:
        d = g.diff(SEASON).abs().dropna()
        return float(d.mean()) if len(d) else np.nan

    return train.sort_values("ds").groupby("unique_id")["y"].apply(one)


def summarize(
    actual: pd.DataFrame, pred: pd.DataFrame, train: pd.DataFrame, quantiles: list[float]
) -> dict[str, float]:
    """Aggregate metrics over all series in a backtest window."""
    merged = actual.merge(pred, on=["unique_id", "ds"], how="inner", suffixes=("", "_pred"))
    if merged.empty:
        raise ValueError("No overlap between actuals and predictions")

    scale = _scale(train).rename("scale")
    merged = merged.join(scale, on="unique_id")
    merged = merged[merged["scale"].notna() & (merged["scale"] > 0)]

    err = merged["y"] - merged["yhat"]
    mase = float((err.abs() / merged["scale"]).mean())
    rmsse = float(np.sqrt(((err**2) / (merged["scale"] ** 2)).mean()))

    metrics = {"mase": round(mase, 4), "rmsse": round(rmsse, 4)}

    pinballs = []
    for q in quantiles:
        col = f"q{int(q * 100)}"
        if col not in merged.columns:
            continue
        diff = merged["y"] - merged[col]
        pinballs.append(np.maximum(q * diff, (q - 1) * diff).mean())
    if pinballs:
        metrics["pinball"] = round(float(np.mean(pinballs)), 4)

    lo, hi = f"q{int(min(quantiles) * 100)}", f"q{int(max(quantiles) * 100)}"
    if lo in merged.columns and hi in merged.columns:
        covered = ((merged["y"] >= merged[lo]) & (merged["y"] <= merged[hi])).mean()
        metrics["coverage"] = round(float(covered), 4)

    return metrics
