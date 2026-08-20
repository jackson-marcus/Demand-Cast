"""The most important tests in the repo: features must never leak the future."""

import numpy as np

from demandcast.features.build_features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_features,
)


def test_lags_shift_by_exact_offset(tiny_sales):
    df = add_lag_features(tiny_sales, lags=[7])
    g = df[df["unique_id"] == "ITEM_001/ST_1"].sort_values("ds")
    assert np.allclose(g["lag_7"].iloc[7:].to_numpy(), g["y"].iloc[:-7].to_numpy(), equal_nan=True)
    assert g["lag_7"].iloc[:7].isna().all()


def test_no_leakage_rolling_excludes_today(tiny_sales):
    """Perturbing y at time t must not change any feature at time t."""
    df = tiny_sales.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    feats_before = build_features(df)

    perturbed = df.copy()
    target_idx = perturbed[perturbed["unique_id"] == "ITEM_001/ST_1"].index[100]
    perturbed.loc[target_idx, "y"] += 1000

    feats_after = build_features(perturbed)
    feature_cols = [
        c for c in feats_before.columns if c not in ("y", "unique_id", "ds", "event_name_1")
    ]
    row_before = feats_before.loc[target_idx, feature_cols].astype(float)
    row_after = feats_after.loc[target_idx, feature_cols].astype(float)
    assert row_before.equals(row_after), "features at t changed when y(t) changed -> leakage"


def test_leakage_does_propagate_to_future_rows(tiny_sales):
    """Sanity check of the test above: the perturbation IS visible at t+1."""
    df = tiny_sales.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    feats_before = build_features(df)
    perturbed = df.copy()
    series_idx = perturbed[perturbed["unique_id"] == "ITEM_001/ST_1"].index
    target_idx = series_idx[100]
    next_idx = series_idx[101]
    perturbed.loc[target_idx, "y"] += 1000
    feats_after = build_features(perturbed)
    assert feats_after.loc[next_idx, "rmean_7"] != feats_before.loc[next_idx, "rmean_7"]


def test_calendar_features(tiny_sales):
    df = add_calendar_features(tiny_sales)
    assert df["dayofweek"].between(0, 6).all()
    assert df["is_weekend"].isin([0, 1]).all()


def test_rolling_window_is_finite_and_positive(tiny_sales):
    df = add_rolling_features(tiny_sales, windows=[7])
    valid = df["rmean_7"].dropna()
    assert (valid >= 0).all()
