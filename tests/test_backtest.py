import pandas as pd

from demandcast.models.backtest import cutoffs, forecast_seasonal_naive


def test_cutoffs_are_ordered_and_leave_room(tiny_sales):
    cuts = cutoffs(tiny_sales, n_windows=3, step=28, horizon=28)
    assert len(cuts) == 3
    assert cuts == sorted(cuts)
    last = tiny_sales["ds"].max()
    for c in cuts:
        assert c + pd.Timedelta(days=28) <= last


def test_seasonal_naive_shape_and_columns(tiny_sales):
    pred = forecast_seasonal_naive(tiny_sales, horizon=14, quantiles=[0.1, 0.5, 0.9])
    assert set(pred.columns) >= {"unique_id", "ds", "yhat", "q10", "q50", "q90"}
    assert len(pred) == tiny_sales["unique_id"].nunique() * 14
    assert (pred["yhat"] >= 0).all()


def test_seasonal_naive_starts_after_training_end(tiny_sales):
    pred = forecast_seasonal_naive(tiny_sales, horizon=7, quantiles=[0.5])
    assert pred["ds"].min() == tiny_sales["ds"].max() + pd.Timedelta(days=1)


def test_seasonal_naive_only_uses_train_data(tiny_sales):
    """Forecast from a truncated panel must not change when future rows change."""
    cutoff = tiny_sales["ds"].max() - pd.Timedelta(days=28)
    train = tiny_sales[tiny_sales["ds"] <= cutoff]

    pred1 = forecast_seasonal_naive(train, horizon=7, quantiles=[0.5])
    corrupted_future = tiny_sales.copy()
    corrupted_future.loc[corrupted_future["ds"] > cutoff, "y"] += 999
    train2 = corrupted_future[corrupted_future["ds"] <= cutoff]
    pred2 = forecast_seasonal_naive(train2, horizon=7, quantiles=[0.5])
    pd.testing.assert_frame_equal(pred1, pred2)
