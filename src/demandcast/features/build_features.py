"""Leakage-safe feature building for tree-based forecasting.

Every feature at time t uses only information available strictly before t
(lags shifted by >= 1) or known in advance (calendar, planned price, SNAP).
The no-leakage property is asserted by tests/test_features.py.
"""

from __future__ import annotations

import pandas as pd

from demandcast.settings import get_config


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    lags = lags or get_config()["features"]["lags"]
    df = df.sort_values(["unique_id", "ds"]).copy()
    g = df.groupby("unique_id", sort=False)["y"]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or get_config()["features"]["rolling_windows"]
    df = df.sort_values(["unique_id", "ds"]).copy()
    # shift(1) BEFORE rolling: the window must end at t-1, never include t.
    shifted = df.groupby("unique_id", sort=False)["y"].shift(1)
    for w in windows:
        df[f"rmean_{w}"] = shifted.groupby(df["unique_id"], sort=False).transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
        df[f"rstd_{w}"] = shifted.groupby(df["unique_id"], sort=False).transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std()
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_calendar_features(add_rolling_features(add_lag_features(df)))
