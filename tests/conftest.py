"""Fixtures: a small synthetic sales panel with known seasonal structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_DAYS = 180
SERIES = ["ITEM_001/ST_1", "ITEM_002/ST_1", "ITEM_003/ST_2"]


@pytest.fixture(scope="session")
def tiny_sales() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=N_DAYS, freq="D")
    frames = []
    for i, uid in enumerate(SERIES):
        base = 5 + 3 * i
        weekly = 1 + 0.4 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
        y = rng.poisson(base * weekly).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": dates,
                    "y": y,
                    "sell_price": 9.99,
                    "event_name_1": None,
                    "snap": (dates.day <= 10).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
