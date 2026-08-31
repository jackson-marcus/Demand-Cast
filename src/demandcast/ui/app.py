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

tab_forecast, tab_order, tab_monitor = st.tabs(
    ["Forecast explorer", "Replenishment", "Accuracy monitor"]
)

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

with tab_order:
    st.markdown(
        "Turns the stored fan into an order-up-to level for one series, then replays "
        "that level against the demand the series actually saw before recommending it."
    )
    if forecasts is None:
        st.info("No stored forecasts - run scripts/refresh_forecasts.py")
    else:
        from demandcast.pipeline.executor import ReplenishmentPlanner, ReplenishmentRequest
        from demandcast.pipeline.steps.fan import HorizonTooShortError, SeriesUnavailableError

        c1, c2, c3, c4 = st.columns(4)
        order_uid = c1.selectbox("Series ", sorted(forecasts["unique_id"].unique()), key="ord_uid")
        lead_time = c2.number_input("Supplier lead time (days)", 0, 27, 7)
        service_level = c3.slider("Target fill rate", 0.50, 0.99, 0.90, 0.01)
        on_hand = c4.number_input("On hand (units)", 0.0, value=0.0, step=1.0)

        planner = ReplenishmentPlanner(forecasts, sales)
        try:
            plan = planner.run(
                ReplenishmentRequest(
                    unique_id=order_uid,
                    on_hand=on_hand,
                    lead_time=int(lead_time),
                    service_level=float(service_level),
                )
            )
        except (SeriesUnavailableError, HorizonTooShortError, ValueError) as exc:
            st.error(str(exc))
        else:
            order, audit, lt = plan["order"], plan["audit"], plan["leadtime"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Order now", f"{order['order_quantity']:.0f} units")
            m2.metric("Order up to", f"{order['order_up_to']:.1f}")
            m3.metric(f"Expected demand ({lt['days']}d)", f"{lt['expected_demand']:.1f}")
            m4.metric("Safety stock", f"{order['safety_stock']:.1f}")

            if plan["status"] == "review":
                st.warning(
                    f"Needs review: replayed against the last {audit['window_days']} days of "
                    f"actual demand this level fills {audit['fill_rate']:.1%}, short of the "
                    f"{service_level:.0%} target ({audit['unmet_units']:.0f} units unmet, "
                    f"{audit['stockout_days']} stockout days)."
                )
            elif plan["status"] == "hold":
                st.success("Hold - the current position already covers the order-up-to level.")
            else:
                st.success(
                    f"Audited fill {audit['fill_rate']:.1%} over the last "
                    f"{audit['window_days']} days of actual demand, mean on-hand "
                    f"{audit['mean_on_hand']:.1f} units."
                )

            st.caption(
                f"Summing the daily P90 instead of adding variances would set the level at "
                f"{order['comonotone_order_up_to']:.1f} - "
                f"{order['comonotone_extra_units']:.1f} extra units carried. "
                f"Spread floored on {plan['spread']['days_floored']} of {lt['days']} days; "
                f"{plan['fan']['crossed_days_repaired']} crossed quantile day(s) repaired. "
                f"Stages: {' -> '.join(plan['stages'])}."
            )
