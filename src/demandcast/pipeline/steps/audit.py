"""Stage 6: replay the level against demand the series actually saw.

Everything upstream of here trusts the forecast. This stage does not: it takes
the order-up-to level the policy just produced and runs a daily-review
base-stock simulation over the last ``audit_days`` of *realised* sales for the
series, then reports the fill rate that level would have achieved.

That is the honest check on a fan we know is under-dispersed. When the audited
fill rate falls short of the requested service level, the plan comes back for
review instead of as a number to act on -- the recommendation is still
returned, but flagged, because the failure mode it catches (demand drifted up
and the band did not follow) is exactly the one that empties a shelf.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def simulate_base_stock(demand: np.ndarray, level: float | np.ndarray, lead_time: int) -> dict:
    """Daily-review, lost-sales base-stock simulation.

    Each day: receive what is due, serve what stock allows (unmet demand is
    lost, not backordered), then order back up to ``level``. Orders are placed
    at the end of the day and land at the start of day ``t + lead_time + 1``,
    which is what makes the protection interval ``lead_time + 1`` days: the
    order placed tonight is the last one that can cover day ``t + lead_time + 1``.
    ``level`` may be a scalar or one level per day.
    """
    demand = np.asarray(demand, dtype=float)
    n = len(demand)
    levels = np.broadcast_to(np.asarray(level, dtype=float), (n,))
    arrivals = np.zeros(n + lead_time + 2)
    on_hand = float(levels[0])
    served = shortfall = 0.0
    on_hand_trace = np.empty(n)
    stockout_days = 0

    for t in range(n):
        on_hand += arrivals[t]
        sold = min(on_hand, demand[t])
        on_hand -= sold
        served += sold
        shortfall += demand[t] - sold
        on_hand_trace[t] = on_hand
        if on_hand <= 1e-9 and demand[t] > 0:
            stockout_days += 1
        pipeline = float(arrivals[t + 1 :].sum())
        arrivals[t + lead_time + 1] += max(0.0, levels[t] - (on_hand + pipeline))

    total = served + shortfall
    return {
        "fill_rate": float(served / total) if total > 0 else 1.0,
        "unmet_units": float(shortfall),
        "stockout_days": stockout_days,
        "mean_on_hand": float(on_hand_trace.mean()),
    }


def run(context: dict[str, Any]) -> dict[str, Any]:
    request = context["request"]
    history: pd.DataFrame = context["history"]
    level = context["order"]["order_up_to"]

    series = history[history["unique_id"] == request.unique_id].sort_values("ds")
    demand = series["y"].to_numpy(dtype=float)[-request.audit_days :]
    if len(demand) == 0:
        raise ValueError(f"no sales history for {request.unique_id!r} to audit against")

    result = simulate_base_stock(demand, level, request.lead_time)
    result["window_days"] = len(demand)
    result["target_fill_rate"] = request.service_level
    result["meets_target"] = bool(result["fill_rate"] >= request.service_level)
    context["audit"] = result
    return context
