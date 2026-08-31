"""Replenishment plan: stage graph, spread calibration, policy, audit gate."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from demandcast.pipeline.dag import STAGES, Stage, topological_order
from demandcast.pipeline.executor import (
    ReplenishmentPlanner,
    ReplenishmentRequest,
    StageContractError,
)
from demandcast.pipeline.steps.audit import simulate_base_stock
from demandcast.pipeline.steps.calibrate import seasonal_residual_sigma
from demandcast.pipeline.steps.fan import HorizonTooShortError, SeriesUnavailableError

UID = "ITEM_001/ST_1"
START = pd.Timestamp("2025-07-01")


def make_fan(days=28, q10=2.0, q50=5.0, q90=9.0, uid=UID):
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": pd.date_range(START, periods=days, freq="D"),
            "q10": np.full(days, float(q10)),
            "q50": np.full(days, float(q50)),
            "q90": np.full(days, float(q90)),
            "yhat": np.full(days, float(q50)),
            "model": "test",
        }
    )


def make_history(values, uid=UID):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": pd.date_range(START - pd.Timedelta(days=len(values)), periods=len(values)),
            "y": values,
        }
    )


def plan(fan=None, history=None, **kwargs):
    fan = make_fan() if fan is None else fan
    history = make_history(np.full(120, 5.0)) if history is None else history
    request = ReplenishmentRequest(unique_id=kwargs.pop("unique_id", UID), **kwargs)
    return ReplenishmentPlanner(fan, history).run(request)


# --- stage graph -------------------------------------------------------------


def test_every_stage_runs_after_all_of_its_dependencies():
    order = topological_order()
    seen: set[str] = set()
    for name in order:
        stage = next(s for s in STAGES if s.name == name)
        assert set(stage.requires) <= seen, f"{name} ran before {set(stage.requires) - seen}"
        seen.add(name)
    assert len(order) == len(STAGES)


def test_order_survives_an_adversarial_declaration_order():
    """Declaration order must not be what makes the sequence valid."""
    reversed_stages = tuple(reversed(STAGES))
    order = topological_order(reversed_stages)
    assert order.index("position") < order.index("policy")
    assert order.index("calibrate") < order.index("leadtime")
    assert order[0] in {"fan", "position"}


def test_cycle_is_reported_not_silently_dropped():
    cyclic = (
        Stage("a", ("b",), "a", ""),
        Stage("b", ("a",), "b", ""),
    )
    with pytest.raises(RuntimeError, match="cycle"):
        topological_order(cyclic)


def test_unknown_dependency_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        topological_order((Stage("a", ("nope",), "a", ""),))


def test_stage_that_skips_its_work_is_caught(monkeypatch):
    import demandcast.pipeline.steps as steps

    monkeypatch.setitem(steps.STAGE_RUNNERS, "leadtime", lambda ctx: ctx)
    with pytest.raises(StageContractError, match="leadtime"):
        plan()


# --- fan repair and spread ---------------------------------------------------


def test_crossed_quantiles_are_repaired_and_counted():
    fan = make_fan()
    fan.loc[3, ["q10", "q50", "q90"]] = [9.0, 5.0, 2.0]  # fully inverted day
    result = plan(fan=fan, lead_time=7)
    assert result["fan"]["crossed_days_repaired"] == 1
    q10, q50, q90 = (result["fan"][k] for k in ("q10", "q50", "q90"))
    assert np.all(q10 <= q50) and np.all(q50 <= q90)
    assert result["fan"]["q90"][3] == pytest.approx(9.0)


def test_seasonal_residual_sigma_recovers_the_generating_sd():
    rng = np.random.default_rng(7)
    y = rng.normal(50.0, 4.0, size=4000)
    assert seasonal_residual_sigma(y) == pytest.approx(4.0, rel=0.06)


def test_sigma_floor_rescues_a_collapsed_band():
    """A fan with no spread would size zero safety stock; history says otherwise."""
    flat = make_fan(q10=5.0, q50=5.0, q90=5.0)
    rng = np.random.default_rng(1)
    volatile = make_history(np.abs(rng.normal(5.0, 3.0, size=120)))

    without = plan(fan=flat, history=volatile, floor_sigma=False)
    with_floor = plan(fan=flat, history=volatile, floor_sigma=True)

    assert without["order"]["safety_stock"] == pytest.approx(0.0)
    assert with_floor["order"]["safety_stock"] > 2.0
    assert with_floor["spread"]["days_floored"] == with_floor["fan"]["days"]


def test_floor_leaves_a_wide_fan_alone():
    wide = make_fan(q10=0.0, q50=5.0, q90=40.0)
    steady = make_history(np.full(120, 5.0))
    result = plan(fan=wide, history=steady)
    assert result["spread"]["days_floored"] == 0
    assert result["spread"]["sigma_floor"] == pytest.approx(0.0)


# --- lead-time aggregation ---------------------------------------------------


@pytest.mark.parametrize("lead_time", [0, 3, 7, 13])
def test_comonotone_spread_is_sqrt_n_larger_than_independent(lead_time):
    """Summing daily P90s scales with n; adding variances scales with sqrt(n)."""
    result = plan(lead_time=lead_time)
    days = result["leadtime"]["days"]
    assert days == lead_time + 1
    ratio = result["leadtime"]["comonotone_sigma"] / result["leadtime"]["sigma"]
    assert ratio == pytest.approx(np.sqrt(days))


def test_protection_window_longer_than_the_stored_fan_is_refused():
    with pytest.raises(HorizonTooShortError, match="protection window needs"):
        plan(fan=make_fan(days=5), lead_time=7)


def test_unknown_series_is_refused():
    with pytest.raises(SeriesUnavailableError):
        plan(unique_id="NOPE/ST_9")


# --- policy ------------------------------------------------------------------


def test_in_transit_units_reduce_the_order_one_for_one():
    base = plan()["order"]["order_quantity"]
    assert plan(in_transit=6.0)["order"]["order_quantity"] == pytest.approx(base - 6.0)
    assert plan(on_hand=6.0)["order"]["order_quantity"] == pytest.approx(base - 6.0)


def test_a_covered_position_orders_nothing_rather_than_a_negative():
    result = plan(on_hand=10_000.0)
    assert result["order"]["order_quantity"] == 0.0
    assert result["status"] == "hold"


def test_higher_service_level_orders_more():
    levels = [plan(service_level=s)["order"]["order_up_to"] for s in (0.5, 0.8, 0.95, 0.99)]
    assert levels == sorted(levels)
    assert levels[0] == pytest.approx(plan()["leadtime"]["expected_demand"])


def test_negative_stock_is_rejected():
    with pytest.raises(ValueError, match="on_hand"):
        plan(on_hand=-1.0)


# --- base-stock simulation and the audit gate --------------------------------


def test_simulation_conserves_demand():
    rng = np.random.default_rng(3)
    demand = rng.poisson(6.0, size=60).astype(float)
    result = simulate_base_stock(demand, level=20.0, lead_time=4)
    served = demand.sum() - result["unmet_units"]
    assert served / demand.sum() == pytest.approx(result["fill_rate"])


def test_fill_rate_is_monotone_in_the_order_up_to_level():
    rng = np.random.default_rng(11)
    demand = rng.poisson(8.0, size=90).astype(float)
    fills = [simulate_base_stock(demand, level=lvl, lead_time=5)["fill_rate"] for lvl in range(0, 120, 10)]
    assert all(b >= a - 1e-12 for a, b in pairwise(fills))
    assert fills[0] < fills[-1]


def test_a_longer_lead_time_cannot_improve_service_at_a_fixed_level():
    rng = np.random.default_rng(5)
    demand = rng.poisson(7.0, size=90).astype(float)
    fills = [simulate_base_stock(demand, level=30.0, lead_time=lt)["fill_rate"] for lt in (0, 3, 7, 14)]
    assert all(b <= a + 1e-12 for a, b in pairwise(fills))


def test_audit_sends_an_under_dispersed_fan_back_for_review():
    """The fan says demand is 5/day; the series has actually been running at 20."""
    result = plan(history=make_history(np.full(120, 20.0)))
    assert result["audit"]["fill_rate"] < 0.9
    assert result["audit"]["unmet_units"] > 0
    assert result["status"] == "review"


def test_review_wins_over_hold():
    """A failing audit must be surfaced even when there is nothing to order today."""
    result = plan(history=make_history(np.full(120, 20.0)), on_hand=10_000.0)
    assert result["order"]["order_quantity"] == 0.0
    assert result["status"] == "review"


def test_audit_passes_when_the_fan_matches_realised_demand():
    result = plan(history=make_history(np.full(120, 5.0)))
    assert result["audit"]["fill_rate"] == pytest.approx(1.0)
    assert result["audit"]["stockout_days"] == 0
    assert result["status"] == "order"


def test_audit_window_is_the_tail_of_history_not_the_whole_series():
    """Old demand must not dilute a recent shift."""
    history = make_history(np.concatenate([np.full(200, 1.0), np.full(28, 30.0)]))
    assert plan(history=history, audit_days=28)["status"] == "review"
    assert plan(history=history, audit_days=228)["audit"]["window_days"] == 228
