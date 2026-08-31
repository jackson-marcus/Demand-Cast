"""Stage 1: pull the stored quantile fan for the protection window.

The protection interval of a daily-review order-up-to policy is
``lead_time + 1`` days: whatever you order today lands in ``lead_time`` days,
and it has to cover demand until the *next* order lands. So this stage takes
the first ``lead_time + 1`` days of the stored forecast for the series.

It also repairs quantile crossing. The recursive LightGBM forecaster fits one
model per quantile independently, so nothing forces q10 <= q50 <= q90; on the
checked-in forecast that happens on a small number of rows, and a crossed row
would hand ``calibrate`` a negative band width. Sorting the three values per
day is the standard fix and is reported back rather than done silently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

QUANTILE_COLUMNS = ("q10", "q50", "q90")


class SeriesUnavailableError(LookupError):
    """No stored forecast for the requested series."""


class HorizonTooShortError(ValueError):
    """The stored forecast does not reach the end of the protection window."""


def run(context: dict[str, Any]) -> dict[str, Any]:
    request = context["request"]
    frame: pd.DataFrame = context["forecasts"]

    missing = [c for c in QUANTILE_COLUMNS if c not in frame.columns]
    if missing:
        raise HorizonTooShortError(f"stored forecast is missing quantile column(s) {missing}")

    part = frame[frame["unique_id"] == request.unique_id].sort_values("ds")
    if part.empty:
        raise SeriesUnavailableError(f"no stored forecast for series {request.unique_id!r}")

    needed = request.lead_time + 1
    if len(part) < needed:
        raise HorizonTooShortError(
            f"{request.unique_id}: protection window needs {needed} forecast days "
            f"(lead time {request.lead_time} + 1), stored forecast has {len(part)}"
        )

    window = part.head(needed)
    raw = np.vstack([window[c].to_numpy(dtype=float) for c in QUANTILE_COLUMNS])
    repaired = np.sort(raw, axis=0)
    crossed_days = int((repaired != raw).any(axis=0).sum())

    context["fan"] = {
        "unique_id": request.unique_id,
        "days": needed,
        "start": pd.Timestamp(window["ds"].iloc[0]),
        "end": pd.Timestamp(window["ds"].iloc[-1]),
        "q10": repaired[0],
        "q50": repaired[1],
        "q90": repaired[2],
        "crossed_days_repaired": crossed_days,
    }
    return context
