"""Transformer Chain Architecture - Concrete Transformation Steps.

Individual transformer objects adhering strictly to the fit/transform contract.
"""

from __future__ import annotations

import pandas as pd


class CalendarFeatureTransformer:
    """Extract temporal calendar features from datetime timestamp."""

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> CalendarFeatureTransformer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ds = pd.to_datetime(df["ds"])
        df["dayofweek"] = ds.dt.dayofweek
        df["month"] = ds.dt.month
        df["day"] = ds.dt.day
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


class LagFeatureTransformer:
    """Construct shift-based lag features guaranteed to prevent future target leakage."""

    def __init__(self, lags: list[int] | None = None) -> None:
        self.lags = lags or [7, 14, 28]

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> LagFeatureTransformer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unique_id", "ds"]).copy()
        g = df.groupby("unique_id", sort=False)["y"]
        for lag in self.lags:
            df[f"lag_{lag}"] = g.shift(lag)
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


class RollingStatTransformer:
    """Compute rolling window aggregations strictly shifted before the forecast timestamp."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [7, 28]

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> RollingStatTransformer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unique_id", "ds"]).copy()
        shifted = df.groupby("unique_id", sort=False)["y"].shift(1)
        for w in self.windows:
            df[f"rmean_{w}"] = shifted.groupby(df["unique_id"], sort=False).transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
            df[f"rstd_{w}"] = shifted.groupby(df["unique_id"], sort=False).transform(
                lambda s, w=w: s.rolling(w, min_periods=2).std()
            )
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


class OutlierClipperTransformer:
    """Clip anomalous demand spikes using interquartile range bounds learned at fit time."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> OutlierClipperTransformer:
        self.bounds_ = {}
        for uid, group in df.groupby("unique_id", sort=False):
            low = float(group["y"].quantile(self.lower_quantile))
            high = float(group["y"].quantile(self.upper_quantile))
            self.bounds_[str(uid)] = (low, high)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.bounds_:
            return df.copy()
        df = df.copy()
        for uid, (low, high) in self.bounds_.items():
            mask = df["unique_id"] == uid
            df.loc[mask, "y"] = df.loc[mask, "y"].clip(lower=low, upper=high)
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
