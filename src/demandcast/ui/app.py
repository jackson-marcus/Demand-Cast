"""Streamlit planning dashboard: fan charts, leaderboard, accuracy monitor."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandcast.models.backtest import load_sales
from demandcast.settings import get_config, resolve_path

st.set_page_config(page_title="demandcast", page_icon="📈", layout="wide")
st.title("📈 demandcast")
st.caption("Probabilistic retail demand forecasts with rolling-origin backtesting")


@st.cache_data
def _sales() -> pd.DataFrame:
    return load_sales()


@st.cache_data
def _forecasts() -> pd.DataFrame | None:
    path = resolve_path(get_config()["data"]["forecasts_dir"]) / "latest.parquet"
    return pd.read_parquet(path) if path.exists() else None


sales = _sales()
forecasts = _forecasts()

tab_forecast, tab_monitor = st.tabs(["Forecast explorer", "Accuracy monitor"])

with tab_forecast:
    uid = st.selectbox("Series", sorted(sales["unique_id"].unique()))
    history_days = st.slider("History shown (days)", 60, 365, 120)

    hist = sales[sales["unique_id"] == uid].sort_values("ds").tail(history_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="actual", line={"color": "#636EFA"}))

    if forecasts is not None and (fc := forecasts[forecasts["unique_id"] == uid]).shape[0]:
        fc = fc.sort_values("ds")
        if {"q10", "q90"}.issubset(fc.columns):
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                    y=pd.concat([fc["q90"], fc["q10"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(239,85,59,0.15)",
                    line={"width": 0},
                    name="P10-P90",
                )
            )
        fig.add_trace(
            go.Scatter(x=fc["ds"], y=fc["yhat"], name="forecast", line={"color": "#EF553B"})
        )
    else:
        st.info("No stored forecasts — run scripts/refresh_forecasts.py")

    fig.update_layout(height=450, margin={"l": 20, "r": 20, "t": 30, "b": 20})
    st.plotly_chart(fig, use_container_width=True)

with tab_monitor:
    st.markdown(
        "Simulates 'new actuals arrive': compares stored forecasts against the "
        "most recent actuals and flags degradation."
    )
    if forecasts is None:
        st.info("No stored forecasts — run scripts/refresh_forecasts.py")
    else:
        from demandcast.monitoring.accuracy_tracker import check_accuracy

        degrade = st.slider("Simulate demand shift (x actual demand)", 1.0, 3.0, 1.0, 0.1)
        horizon = get_config()["forecast"]["horizon"]
        last_train_day = sales["ds"].max() - pd.Timedelta(days=horizon)
        train = sales[sales["ds"] <= last_train_day]
        actuals = sales[sales["ds"] > last_train_day].copy()
        actuals["y"] = actuals["y"] * degrade

        try:
            status = check_accuracy(actuals, forecasts, train)
        except ValueError:
            st.warning("Stored forecasts do not overlap the held-out actuals window.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Rolling MASE", f"{status.mase:.3f}")
            col2.metric("Alert threshold", f"{status.threshold:.2f}")
            col3.metric("Points compared", status.n_points)
            if status.degraded:
                st.error("🚨 Forecast accuracy degraded — retrain or investigate demand shift.")
            else:
                st.success("✅ Forecast accuracy within tolerance.")
