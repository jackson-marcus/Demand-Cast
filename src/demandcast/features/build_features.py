"""Feature building layer powered by the Transformer Chain Architecture."""

from __future__ import annotations

import pandas as pd

from demandcast.pipeline.chain import build_default_feature_chain
from demandcast.pipeline.transformers import (
    CalendarFeatureTransformer,
    LagFeatureTransformer,
    RollingStatTransformer,
)


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    return LagFeatureTransformer(lags=lags).transform(df)


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    return RollingStatTransformer(windows=windows).transform(df)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    return CalendarFeatureTransformer().transform(df)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all leak-free features via the composable TransformerChain."""
    return build_default_feature_chain().transform(df)
