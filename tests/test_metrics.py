import numpy as np
import pandas as pd
import pytest

from demandcast.evaluation.metrics import summarize


def _split(tiny_sales, horizon=14):
    cutoff = tiny_sales["ds"].max() - pd.Timedelta(days=horizon)
    return tiny_sales[tiny_sales["ds"] <= cutoff], tiny_sales[tiny_sales["ds"] > cutoff]


def test_perfect_forecast_scores_zero(tiny_sales):
    train, actual = _split(tiny_sales)
    pred = actual.rename(columns={"y": "yhat"})[["unique_id", "ds", "yhat"]].copy()
    pred["q10"] = pred["yhat"] - 1
    pred["q50"] = pred["yhat"]
    pred["q90"] = pred["yhat"] + 1
    m = summarize(actual, pred, train, [0.1, 0.5, 0.9])
    assert m["mase"] == 0.0
    assert m["rmsse"] == 0.0
    assert m["coverage"] == 1.0


def test_worse_forecast_scores_higher(tiny_sales):
    train, actual = _split(tiny_sales)
    good = actual.rename(columns={"y": "yhat"})[["unique_id", "ds", "yhat"]].copy()
    bad = good.copy()
    bad["yhat"] = bad["yhat"] + 10
    m_good = summarize(actual, good, train, [0.5])
    m_bad = summarize(actual, bad, train, [0.5])
    assert m_bad["mase"] > m_good["mase"]


def test_no_overlap_raises(tiny_sales):
    train, actual = _split(tiny_sales)
    pred = actual.rename(columns={"y": "yhat"})[["unique_id", "ds", "yhat"]].copy()
    pred["ds"] = pred["ds"] + pd.Timedelta(days=999)
    with pytest.raises(ValueError, match="No overlap"):
        summarize(actual, pred, train, [0.5])


def test_coverage_reflects_interval_width(tiny_sales):
    train, actual = _split(tiny_sales)
    pred = actual.rename(columns={"y": "yhat"})[["unique_id", "ds", "yhat"]].copy()
    pred["q10"] = 0.0
    pred["q90"] = 10_000.0
    wide = summarize(actual, pred, train, [0.1, 0.9])["coverage"]
    pred["q90"] = 0.0001
    narrow = summarize(actual, pred, train, [0.1, 0.9])["coverage"]
    assert wide == 1.0
    assert narrow < wide


def test_mase_scale_is_seasonal(tiny_sales):
    """A seasonal-naive forecast on strongly weekly data should score near 1."""
    train, actual = _split(tiny_sales, horizon=14)
    pred_rows = []
    for uid, g in train.groupby("unique_id"):
        g = g.sort_values("ds")
        future = pd.date_range(g["ds"].max() + pd.Timedelta(days=1), periods=14, freq="D")
        last_week = g.tail(7).set_index(g.tail(7)["ds"].dt.dayofweek)["y"]
        for ds in future:
            pred_rows.append(
                {"unique_id": uid, "ds": ds, "yhat": float(last_week.get(ds.dayofweek, 0))}
            )
    pred = pd.DataFrame(pred_rows)
    m = summarize(actual, pred, train, [0.5])
    assert 0.3 < m["mase"] < 3.0
    assert np.isfinite(m["rmsse"])
