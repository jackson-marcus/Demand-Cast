"""Generate synthetic retail sales in the canonical long format.

Realistic structure: weekly + yearly seasonality, item-level trend, price
elasticity, promotion spikes, SNAP effects, intermittency for slow movers,
and Poisson noise. Lets the whole pipeline run without the M5 download.

Usage:
    uv run python scripts/make_synthetic.py [--items 40] [--stores 2] [--days 730]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandcast.settings import get_config, resolve_path


def generate(items: int, stores: int, days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    frames = []
    for s in range(stores):
        store_id = f"ST_{s + 1}"
        store_scale = rng.uniform(0.8, 1.3)
        for i in range(items):
            item_id = f"ITEM_{i + 1:03d}"
            base = rng.lognormal(1.6, 0.7) * store_scale
            trend = rng.normal(0.0, 0.0004)
            weekly = 1 + 0.35 * np.sin(2 * np.pi * (dow + rng.integers(0, 7)) / 7)
            yearly = 1 + 0.25 * np.sin(2 * np.pi * (doy + rng.integers(0, 365)) / 365)

            price = np.round(rng.uniform(2, 25) * (1 + 0.1 * np.sin(2 * np.pi * doy / 90)), 2)
            promo = rng.random(days) < 0.05
            price = np.where(promo, price * 0.7, price)
            elasticity = np.exp(-0.4 * (price / price.mean() - 1))

            snap = ((dates.day <= 10) & (rng.random(days) < 0.6)).astype(int)
            snap_lift = 1 + 0.15 * snap

            mean = base * (1 + trend * np.arange(days)) * weekly * yearly * elasticity * snap_lift
            mean *= np.where(promo, 1.8, 1.0)
            y = rng.poisson(np.maximum(mean, 0.05)).astype(float)
            if base < 3:  # slow movers: extra zero-inflation (intermittent demand)
                y *= rng.random(days) > 0.25

            frames.append(
                pd.DataFrame(
                    {
                        "unique_id": f"{item_id}/{store_id}",
                        "ds": dates,
                        "y": y,
                        "sell_price": price,
                        "event_name_1": np.where(promo, "Promo", None),
                        "snap": snap,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--stores", type=int, default=2)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.items, args.stores, args.days, args.seed)
    raw_dir = resolve_path(get_config()["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "synthetic_sales.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} rows, {df['unique_id'].nunique()} series -> {out}")


if __name__ == "__main__":
    main()
