import pandas as pd
import pytest
from fastapi.testclient import TestClient

import demandcast.api.routes as routes
from demandcast.api.main import create_app


@pytest.fixture()
def client(tiny_sales, tmp_path, monkeypatch):
    """API wired to a tiny forecast fan and the matching sales history."""
    horizon_start = tiny_sales["ds"].max() + pd.Timedelta(days=1)
    rows = []
    for uid in tiny_sales["unique_id"].unique():
        for i in range(28):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": horizon_start + pd.Timedelta(days=i),
                    "yhat": 5.0,
                    "q10": 2.0,
                    "q50": 5.0,
                    "q90": 9.0,
                    "model": "lgbm",
                }
            )
    fdir = tmp_path / "forecasts"
    fdir.mkdir()
    pd.DataFrame(rows).to_parquet(fdir / "latest.parquet", index=False)

    pdir = tmp_path / "processed"
    pdir.mkdir()
    tiny_sales.to_parquet(pdir / "sales.parquet", index=False)

    from demandcast.settings import get_config

    cfg = get_config()
    original = cfg["data"]["forecasts_dir"], cfg["data"]["processed_dir"]
    cfg["data"]["forecasts_dir"] = str(fdir)
    cfg["data"]["processed_dir"] = str(pdir)
    routes.invalidate_cache()
    yield TestClient(create_app())
    cfg["data"]["forecasts_dir"], cfg["data"]["processed_dir"] = original
    routes.invalidate_cache()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_series_lists_all(client):
    r = client.get("/series")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_forecast_returns_fan(client):
    r = client.get("/forecast", params={"unique_id": "ITEM_001/ST_1", "h": 14})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == 14
    assert body["model"] == "lgbm"
    p = body["points"][0]
    assert p["q10"] <= p["q50"] <= p["q90"]


def test_forecast_unknown_series_404(client):
    r = client.get("/forecast", params={"unique_id": "NOPE/ST_9"})
    assert r.status_code == 404


def test_replenish_orders_up_to_demand_plus_safety(client):
    r = client.get(
        "/replenish",
        params={"unique_id": "ITEM_001/ST_1", "on_hand": 4, "lead_time": 7, "service_level": 0.9},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["protection_days"] == 8
    assert body["expected_demand"] == pytest.approx(8 * 5.0)
    assert body["order_up_to"] == pytest.approx(body["expected_demand"] + body["safety_stock"])
    assert body["order_quantity"] == pytest.approx(body["order_up_to"] - 4.0)
    assert body["stages"][-1] == "audit"


def test_replenish_reports_what_the_p90_shortcut_would_cost(client):
    body = client.get("/replenish", params={"unique_id": "ITEM_001/ST_1"}).json()
    assert body["comonotone_order_up_to"] > body["order_up_to"]
    assert body["comonotone_extra_units"] == pytest.approx(
        body["comonotone_order_up_to"] - body["order_up_to"]
    )


def test_replenish_flags_a_series_whose_demand_outgrew_the_fan(client):
    """ITEM_003/ST_2 runs at ~11 units/day; the stub fan says 5."""
    body = client.get("/replenish", params={"unique_id": "ITEM_003/ST_2"}).json()
    assert body["audit"]["fill_rate"] < body["audit"]["target_fill_rate"]
    assert body["audit"]["stockout_days"] > 0
    assert body["status"] == "review"


def test_replenish_unknown_series_404(client):
    r = client.get("/replenish", params={"unique_id": "NOPE/ST_9"})
    assert r.status_code == 404


def test_replenish_rejects_a_lead_time_the_fan_cannot_cover(client):
    r = client.get("/replenish", params={"unique_id": "ITEM_001/ST_1", "lead_time": 40})
    assert r.status_code == 422
