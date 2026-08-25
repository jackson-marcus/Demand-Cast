"""Unit tests for the Transformer Chain Architecture."""

import numpy as np
import pandas as pd

from demandcast.pipeline.chain import build_default_feature_chain
from demandcast.pipeline.transformers import (
    CalendarFeatureTransformer,
    LagFeatureTransformer,
    OutlierClipperTransformer,
)


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df1 = pd.DataFrame({"unique_id": "SKU_A", "ds": dates, "y": np.arange(1, 31, dtype=float)})
    df2 = pd.DataFrame({"unique_id": "SKU_B", "ds": dates, "y": np.arange(10, 40, dtype=float)})
    return pd.concat([df1, df2], ignore_index=True)


def test_calendar_transformer():
    df = _sample_df()
    trans = CalendarFeatureTransformer()
    res = trans.fit_transform(df)

    assert "dayofweek" in res.columns
    assert "month" in res.columns
    assert "day" in res.columns
    assert "is_weekend" in res.columns
    assert set(res["is_weekend"].unique()).issubset({0, 1})


def test_lag_transformer_no_leakage():
    df = _sample_df()
    trans = LagFeatureTransformer(lags=[7, 14])
    res = trans.fit_transform(df)

    assert "lag_7" in res.columns
    assert "lag_14" in res.columns

    # Verify lag 7 at index 7 equals y at index 0 for SKU_A
    sku_a = res[res["unique_id"] == "SKU_A"].reset_index(drop=True)
    assert np.isnan(sku_a.loc[0:6, "lag_7"]).all()
    assert sku_a.loc[7, "lag_7"] == sku_a.loc[0, "y"]


def test_outlier_clipper_transformer():
    df = _sample_df()
    df.loc[df["unique_id"] == "SKU_A", "y"].iloc[15] = 1000.0  # Spurious spike
    clipper = OutlierClipperTransformer(lower_quantile=0.05, upper_quantile=0.95)
    clipper.fit(df)
    res = clipper.transform(df)

    assert res["y"].max() < 1000.0


def test_transformer_chain_composition():
    df = _sample_df()
    chain = build_default_feature_chain()
    res = chain.fit_transform(df)

    assert "dayofweek" in res.columns
    assert "lag_7" in res.columns
    assert "rmean_7" in res.columns
    assert "rstd_7" in res.columns
    assert len(res) == len(df)
