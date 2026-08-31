"""Stage 3: derive a usable daily demand spread from the fan.

An order-up-to level needs a standard deviation, and all the forecast gives us
is a P10-P90 band. Under a normal working assumption the band is
``2 * z(0.9) * sigma`` wide, which inverts to :func:`sigma_from_band`.

That would be the end of it if the fan were calibrated. It is not: on a 28-day
holdout the checked-in LightGBM fan's P10-P90 band contains 70.7% of realised
demand against a nominal 80% (``scripts/leadtime_study.py``). Recursive
forecasting is the reason -- each step feeds its own point prediction back in
as if it were a fact, so the quantile models never see the uncertainty they
have already accumulated, and the band stops widening with horizon.

So the implied sigma is floored at the series' own recent seasonal-naive
residual volatility, which is an assumption-free read of how much this series
actually moves week to week. ``var(y_t - y_{t-7}) = 2 * var(y)`` for a
stationary series, hence the ``sqrt(2)``.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from demandcast.evaluation.metrics import SEASON

Z90 = NormalDist().inv_cdf(0.9)


def sigma_from_band(q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    """Daily sigma implied by a P10-P90 band, clipped at zero."""
    return np.maximum((np.asarray(q90) - np.asarray(q10)) / (2.0 * Z90), 0.0)


def seasonal_residual_sigma(y: np.ndarray, season: int = SEASON) -> float:
    """Demand sigma implied by the seasonal-naive residuals of ``y``."""
    y = np.asarray(y, dtype=float)
    if len(y) <= season:
        return 0.0
    residuals = y[season:] - y[:-season]
    return float(np.std(residuals) / np.sqrt(2.0))


def run(context: dict[str, Any]) -> dict[str, Any]:
    request = context["request"]
    fan = context["fan"]
    history: pd.DataFrame = context["history"]

    implied = sigma_from_band(fan["q10"], fan["q90"])

    series = history[history["unique_id"] == request.unique_id].sort_values("ds")
    recent = series["y"].to_numpy(dtype=float)[-request.sigma_history_days :]
    floor = seasonal_residual_sigma(recent) if request.floor_sigma else 0.0

    sigma = np.maximum(implied, floor)
    context["spread"] = {
        "sigma_daily": sigma,
        "sigma_implied_mean": float(implied.mean()),
        "sigma_floor": floor,
        "days_floored": int((implied < floor).sum()),
        "history_days_used": len(recent),
    }
    return context
