"""API routes: /forecast, /series, /health.

Forecasts are precomputed by scripts/refresh_forecasts.py (training on-the-fly
per request would be neither fast nor reproducible) and served from parquet.
"""

from __future__ import annotations

import functools
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

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


def invalidate_cache() -> None:
    _forecasts.cache_clear()


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
