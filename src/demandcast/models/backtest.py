"""Forecasters and the rolling-origin backtest harness.

The canonical long sales frame (see ``demandcast.data.prepare``) has columns
``unique_id, ds, y, sell_price, event_name_1, snap``. Every forecaster here
consumes that frame plus a ``horizon`` (days) and a list of ``quantiles`` and
returns a tidy prediction frame with columns::

    unique_id, ds, yhat, q10, q50, q90, ...

one row per (series, future day). Backtesting replays history from a set of
rolling-origin ``cutoffs``: train on everything up to the cutoff, forecast the
next ``horizon`` days, score against the held-out actuals.

Run the harness with::

    uv run python -m demandcast.models.backtest
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from demandcast.evaluation.metrics import SEASON, summarize
from demandcast.pipeline.chain import build_default_feature_chain
from demandcast.settings import get_config, resolve_path

logger = logging.getLogger(__name__)

# A forecaster maps (long sales frame, horizon days, quantiles) -> prediction frame.
Forecaster = Callable[[pd.DataFrame, int, Sequence[float]], pd.DataFrame]


def _quantile_col(q: float) -> str:
    """Column name for a quantile, matching ``evaluation.metrics`` (q=0.1 -> 'q10')."""
    return f"q{round(q * 100)}"


def load_sales(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical processed sales panel written by ``demandcast.data.prepare``.

    Defaults to ``<processed_dir>/sales.parquet`` from the config.
    """
    if path is None:
        path = resolve_path(get_config()["data"]["processed_dir"]) / "sales.parquet"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No sales panel at {path}. Build it with `python -m demandcast.data.prepare` "
            "(or `make data`)."
        )
    df = pd.read_parquet(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def cutoffs(df: pd.DataFrame, n_windows: int, step: int, horizon: int) -> list[pd.Timestamp]:
    """Rolling-origin backtest cutoff dates, oldest first.

    The most recent cutoff leaves exactly ``horizon`` days of actuals to score
    against; earlier cutoffs step back by ``step`` days each. Every returned
    cutoff ``c`` satisfies ``c + horizon days <= last observed day``.
    """
    last = pd.Timestamp(df["ds"].max())
    newest = last - pd.Timedelta(days=horizon)
    cuts = [newest - pd.Timedelta(days=step * i) for i in range(n_windows)]
    return sorted(cuts)


def forecast_seasonal_naive(
    df: pd.DataFrame, horizon: int, quantiles: Sequence[float]
) -> pd.DataFrame:
    """Weekly seasonal-naive forecaster: repeat each weekday's last observed value.

    Point forecast for a future day equals the value observed on the same
    weekday of the final training week. Quantile bands are the empirical
    quantiles of the in-sample seasonal-naive residuals per series, so the
    forecast is deterministic and depends only on the training data passed in.
    """
    quantiles = list(quantiles)
    q_cols = {q: _quantile_col(q) for q in quantiles}
    frames: list[pd.DataFrame] = []

    for uid, group in df.sort_values("ds").groupby("unique_id", sort=True):
        y = group["y"].to_numpy(dtype=float)
        last_date = pd.Timestamp(group["ds"].max())
        season = min(SEASON, len(y))
        last_week = y[-season:]

        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        steps = np.arange(horizon)
        yhat = last_week[steps % season]

        # In-sample seasonal-naive residuals drive the predictive spread.
        resid = y[SEASON:] - y[:-SEASON] if len(y) > SEASON else np.array([0.0])

        out = pd.DataFrame({"unique_id": uid, "ds": future_dates, "yhat": np.clip(yhat, 0, None)})
        for q, col in q_cols.items():
            out[col] = np.clip(yhat + float(np.quantile(resid, q)), 0, None)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True)
    return result


def forecast_lgbm(df: pd.DataFrame, horizon: int, quantiles: Sequence[float]) -> pd.DataFrame:
    """Recursive LightGBM forecaster with per-quantile models.

    Trains a point model plus one quantile-objective model per requested
    quantile on the leak-free feature chain, then rolls the horizon forward one
    day at a time, feeding each point prediction back so lag/rolling features
    stay consistent.
    """
    import lightgbm as lgb

    quantiles = list(quantiles)
    q_cols = {q: _quantile_col(q) for q in quantiles}
    cfg = get_config()
    params = dict(cfg["training"]["lgbm"])
    random_state = cfg["training"]["random_state"]

    panel = df[["unique_id", "ds", "y"]].copy()
    chain = build_default_feature_chain()

    train_feats = chain.transform(panel)
    feature_cols = [c for c in train_feats.columns if c not in {"unique_id", "ds", "y"}]
    train_feats = train_feats.dropna(subset=feature_cols)
    if train_feats.empty:
        raise ValueError("Not enough history to build features for the LightGBM forecaster.")

    x_train = train_feats[feature_cols]
    y_train = train_feats["y"]

    point_model = lgb.LGBMRegressor(random_state=random_state, verbosity=-1, **params)
    point_model.fit(x_train, y_train)

    q_models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, random_state=random_state, verbosity=-1, **params
        )
        model.fit(x_train, y_train)
        q_models[q] = model

    last_dates = panel.groupby("unique_id")["ds"].max()
    work = panel.copy()
    predictions: list[pd.DataFrame] = []

    for step in range(1, horizon + 1):
        future = pd.DataFrame(
            {
                "unique_id": last_dates.index,
                "ds": last_dates.to_numpy() + pd.Timedelta(days=step),
                "y": np.nan,
            }
        )
        feats = chain.transform(pd.concat([work, future], ignore_index=True))
        step_rows = feats[feats["y"].isna()].copy()
        x_step = step_rows[feature_cols]

        yhat = np.clip(point_model.predict(x_step), 0, None)
        out = step_rows[["unique_id", "ds"]].copy()
        out["yhat"] = yhat
        for q, col in q_cols.items():
            out[col] = np.clip(q_models[q].predict(x_step), 0, None)
        predictions.append(out)

        # Feed the point forecast back in as the realised value for the next step's lags.
        fed = out[["unique_id", "ds"]].copy()
        fed["y"] = yhat
        work = pd.concat([work, fed], ignore_index=True)

    result = pd.concat(predictions, ignore_index=True)
    return result.sort_values(["unique_id", "ds"]).reset_index(drop=True)


# Registry the batch job and CLI iterate over. Keys are selectable model names.
FORECASTERS: dict[str, Forecaster] = {
    "seasonal_naive": forecast_seasonal_naive,
    "lgbm": forecast_lgbm,
}


def run_backtest(
    df: pd.DataFrame,
    forecaster: Forecaster,
    horizon: int,
    quantiles: Sequence[float],
    n_windows: int,
    step: int,
) -> pd.DataFrame:
    """Replay ``forecaster`` over rolling-origin windows and score each one."""
    rows = []
    for cutoff in cutoffs(df, n_windows=n_windows, step=step, horizon=horizon):
        train = df[df["ds"] <= cutoff]
        horizon_end = cutoff + pd.Timedelta(days=horizon)
        actual = df[(df["ds"] > cutoff) & (df["ds"] <= horizon_end)]
        if train.empty or actual.empty:
            continue
        pred = forecaster(train, horizon, quantiles)
        metrics = summarize(actual, pred, train, list(quantiles))
        rows.append({"cutoff": cutoff, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = get_config()
    df = load_sales()
    horizon = cfg["forecast"]["horizon"]
    quantiles = cfg["forecast"]["quantiles"]
    n_windows = cfg["backtest"]["n_windows"]
    step = cfg["backtest"]["step_size"]

    for name, forecaster in FORECASTERS.items():
        report = run_backtest(df, forecaster, horizon, quantiles, n_windows, step)
        if report.empty:
            logger.warning("%s: no scorable windows", name)
            continue
        means = report.drop(columns=["cutoff"]).mean(numeric_only=True)
        summary = ", ".join(f"{k}={v:.4f}" for k, v in means.items())
        logger.info("%s over %d windows: %s", name, len(report), summary)


if __name__ == "__main__":
    main()
