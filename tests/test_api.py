import pandas as pd
import pytest
from fastapi.testclient import TestClient

import demandcast.api.routes as routes
from demandcast.api.main import create_app


@pytest.fixture()
def client(tiny_sales, tmp_path, monkeypatch):
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

    from demandcast.settings import get_config

    cfg = get_config()
    original = cfg["data"]["forecasts_dir"]
    cfg["data"]["forecasts_dir"] = str(fdir)
    routes.invalidate_cache()
    yield TestClient(create_app())
    cfg["data"]["forecasts_dir"] = original
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
