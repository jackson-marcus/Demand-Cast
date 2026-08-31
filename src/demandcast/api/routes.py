"""API routes: /forecast, /replenish, /series, /health.

Forecasts are precomputed by scripts/refresh_forecasts.py (training on-the-fly
per request would be neither fast nor reproducible) and served from parquet.

/forecast hands back the fan. /replenish answers the question the fan exists
for -- how many units to order today -- by running the staged plan in
demandcast.pipeline over the same parquet plus the realised sales panel.
"""

from __future__ import annotations

import functools
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from demandcast.models.backtest import load_sales
from demandcast.pipeline.executor import ReplenishmentPlanner, ReplenishmentRequest
from demandcast.pipeline.steps.fan import HorizonTooShortError, SeriesUnavailableError
from demandcast.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    q10: float | None = None
    q50: float | None = None
    q90: float | None = None


class ForecastResponse(BaseModel):
    unique_id: str
    model: str
    horizon: int
    points: list[ForecastPoint]


@functools.lru_cache(maxsize=1)
def _forecasts() -> pd.DataFrame:
    path = resolve_path(get_config()["data"]["forecasts_dir"]) / "latest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No forecasts at {path}; run scripts/refresh_forecasts.py")
    return pd.read_parquet(path)


@functools.lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    return load_sales()


def invalidate_cache() -> None:
    _forecasts.cache_clear()
    _history.cache_clear()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/series")
def series() -> list[str]:
    try:
        return sorted(_forecasts()["unique_id"].unique().tolist())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/forecast", response_model=ForecastResponse)
def forecast(
    unique_id: str = Query(..., description="e.g. ITEM_001/ST_1"),
    h: int = Query(default=28, ge=1, le=28),
) -> ForecastResponse:
    try:
        df = _forecasts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    part = df[df["unique_id"] == unique_id].sort_values("ds").head(h)
    if part.empty:
        raise HTTPException(status_code=404, detail=f"Unknown series: {unique_id}")

    points = [
        ForecastPoint(
            ds=str(pd.Timestamp(row.ds).date()),
            yhat=round(float(row.yhat), 3),
            q10=round(float(row.q10), 3) if "q10" in part.columns else None,
            q50=round(float(row.q50), 3) if "q50" in part.columns else None,
            q90=round(float(row.q90), 3) if "q90" in part.columns else None,
        )
        for row in part.itertuples()
    ]
    return ForecastResponse(
        unique_id=unique_id,
        model=str(part["model"].iloc[0]) if "model" in part.columns else "unknown",
        horizon=len(points),
        points=points,
    )


class SpreadInfo(BaseModel):
    sigma_daily_mean: float
    sigma_floor: float
    days_floored: int
    crossed_days_repaired: int


class AuditInfo(BaseModel):
    window_days: int
    fill_rate: float
    target_fill_rate: float
    stockout_days: int
    unmet_units: float
    mean_on_hand: float
    meets_target: bool


class ReplenishmentResponse(BaseModel):
    unique_id: str
    status: str
    order_quantity: float
    order_up_to: float
    inventory_position: float
    protection_days: int
    expected_demand: float
    safety_stock: float
    comonotone_order_up_to: float
    comonotone_extra_units: float
    spread: SpreadInfo
    audit: AuditInfo
    stages: list[str]


@router.get("/replenish", response_model=ReplenishmentResponse)
def replenish(
    unique_id: str = Query(..., description="e.g. ITEM_001/ST_1"),
    on_hand: float = Query(default=0.0, ge=0, description="units physically in stock"),
    in_transit: float = Query(default=0.0, ge=0, description="units already ordered"),
    lead_time: int = Query(default=7, ge=0, le=27, description="supplier lead time in days"),
    service_level: float = Query(default=0.9, ge=0.5, lt=1.0),
) -> ReplenishmentResponse:
    """How many units to order today, and whether that level survives an audit."""
    try:
        planner = ReplenishmentPlanner(_forecasts(), _history())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    request = ReplenishmentRequest(
        unique_id=unique_id,
        on_hand=on_hand,
        in_transit=in_transit,
        lead_time=lead_time,
        service_level=service_level,
    )
    try:
        result = planner.run(request)
    except SeriesUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HorizonTooShortError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    order, spread, audit = result["order"], result["spread"], result["audit"]
    return ReplenishmentResponse(
        unique_id=unique_id,
        status=result["status"],
        order_quantity=round(order["order_quantity"], 2),
        order_up_to=round(order["order_up_to"], 2),
        inventory_position=result["position"]["inventory_position"],
        protection_days=result["leadtime"]["days"],
        expected_demand=round(result["leadtime"]["expected_demand"], 2),
        safety_stock=round(order["safety_stock"], 2),
        comonotone_order_up_to=round(order["comonotone_order_up_to"], 2),
        comonotone_extra_units=round(order["comonotone_extra_units"], 2),
        spread=SpreadInfo(
            sigma_daily_mean=round(float(spread["sigma_daily"].mean()), 3),
            sigma_floor=round(spread["sigma_floor"], 3),
            days_floored=spread["days_floored"],
            crossed_days_repaired=result["fan"]["crossed_days_repaired"],
        ),
        audit=AuditInfo(
            window_days=audit["window_days"],
            fill_rate=round(audit["fill_rate"], 4),
            target_fill_rate=audit["target_fill_rate"],
            stockout_days=audit["stockout_days"],
            unmet_units=round(audit["unmet_units"], 2),
            mean_on_hand=round(audit["mean_on_hand"], 2),
            meets_target=audit["meets_target"],
        ),
        stages=result["stages"],
    )
