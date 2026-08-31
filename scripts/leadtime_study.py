"""Measure what lead-time aggregation costs, and how honest the fan's band is.

Holds out the last `horizon` days of the sales panel, refits the LightGBM
quantile forecaster on the rest, then drives the same base-stock simulator the
`audit` stage uses with three ways of turning the daily fan into an order-up-to
level:

  quantile_sum   sum the daily P90            (comonotone: the planner shortcut)
  variance_sum   sum q50, add sigmas in quadrature       (independent daily demand)
  floored        variance_sum, but each day's sigma floored at the series'
                 own seasonal-naive residual volatility  (what /replenish does)

Also reports the empirical coverage of the P10-P90 band, and how many of the
live stored series the /replenish audit gate would send back for review.

Usage:
    uv run python scripts/leadtime_study.py [--lead-time 7] [--service-level 0.9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandcast.models.backtest import forecast_lgbm, load_sales
from demandcast.pipeline.executor import ReplenishmentPlanner, ReplenishmentRequest
from demandcast.pipeline.steps.audit import simulate_base_stock
from demandcast.pipeline.steps.calibrate import seasonal_residual_sigma, sigma_from_band
from demandcast.settings import get_config, resolve_path


def levels(fan: pd.DataFrame, how: str, z: float, floor: float, lead_time: int) -> np.ndarray:
    q50 = fan["q50"].to_numpy(dtype=float)
    sigma = sigma_from_band(fan["q10"].to_numpy(dtype=float), fan["q90"].to_numpy(dtype=float))
    if how == "floored":
        sigma = np.maximum(sigma, floor)
    n = len(q50)
    out = np.empty(n)
    for t in range(n):
        window = slice(t, min(t + lead_time + 1, n))
        if how == "quantile_sum":
            out[t] = fan["q90"].to_numpy(dtype=float)[window].sum()
        else:
            out[t] = q50[window].sum() + z * float(np.sqrt(np.square(sigma[window]).sum()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-time", type=int, default=7)
    parser.add_argument("--service-level", type=float, default=0.9)
    args = parser.parse_args()

    cfg = get_config()
    horizon = cfg["forecast"]["horizon"]
    quantiles = cfg["forecast"]["quantiles"]
    z = NormalDist().inv_cdf(args.service_level)

    sales = load_sales()
    cutoff = pd.Timestamp(sales["ds"].max()) - pd.Timedelta(days=horizon)
    train = sales[sales["ds"] <= cutoff]
    actual = sales[sales["ds"] > cutoff]
    print(f"holdout cutoff {cutoff.date()}  train={len(train):,} rows  holdout={len(actual):,} rows")

    pred = forecast_lgbm(train, horizon, quantiles)
    merged = actual.merge(pred, on=["unique_id", "ds"])
    coverage = float(((merged["y"] >= merged["q10"]) & (merged["y"] <= merged["q90"])).mean())
    nominal = max(quantiles) - min(quantiles)
    crossed = int(((pred["q10"] > pred["q50"]) | (pred["q50"] > pred["q90"])).sum())
    print(f"P10-P90 band covers {coverage:.4f} of held-out actuals (nominal {nominal:.2f})")
    print(f"quantile crossing on {crossed} of {len(pred)} forecast rows")

    rows = []
    for uid, group in sales.groupby("unique_id"):
        fan = pred[pred["unique_id"] == uid].sort_values("ds")
        held = group[group["ds"] > cutoff].sort_values("ds")
        if len(fan) != horizon or len(held) != horizon:
            continue
        history = group[group["ds"] <= cutoff]["y"].to_numpy(dtype=float)[-63:]
        floor = seasonal_residual_sigma(history)
        demand = held["y"].to_numpy(dtype=float)
        for how in ("quantile_sum", "variance_sum", "floored"):
            result = simulate_base_stock(
                demand, levels(fan, how, z, floor, args.lead_time), args.lead_time
            )
            rows.append({"unique_id": uid, "policy": how, **result})

    report = pd.DataFrame(rows)
    print(
        f"\nbase-stock replay: {report['unique_id'].nunique()} series, "
        f"lead time {args.lead_time}d, target fill {args.service_level:.0%}"
    )
    print(f"{'policy':>13} {'fill':>8} {'mean on-hand':>13} {'stockout d':>11} {'below target':>13}")
    summary = {}
    for how in ("quantile_sum", "variance_sum", "floored"):
        g = report[report["policy"] == how]
        summary[how] = g["mean_on_hand"].mean()
        print(
            f"{how:>13} {g['fill_rate'].mean():>8.4f} {g['mean_on_hand'].mean():>13.2f} "
            f"{int(g['stockout_days'].sum()):>11d} "
            f"{int((g['fill_rate'] < args.service_level).sum()):>13d}"
        )
    q, v, f = summary["quantile_sum"], summary["variance_sum"], summary["floored"]
    print(f"\nquantile_sum holds {q / v - 1:+.1%} inventory vs variance_sum, {q / f - 1:+.1%} vs floored")

    live = resolve_path(cfg["data"]["forecasts_dir"]) / "latest.parquet"
    if live.exists():
        planner = ReplenishmentPlanner(pd.read_parquet(live), sales)
        flagged = [
            uid
            for uid in sorted(pd.read_parquet(live)["unique_id"].unique())
            if planner.run(
                ReplenishmentRequest(
                    unique_id=uid, lead_time=args.lead_time, service_level=args.service_level
                )
            )["status"]
            == "review"
        ]
        print(f"\n/replenish audit gate flags {len(flagged)} of {len(planner.forecasts['unique_id'].unique())} live series for review")


if __name__ == "__main__":
    main()
