"""Prepare the M5 dataset: wide -> long, join calendar + prices, subset.

Produces the canonical long frame the whole project uses:
    unique_id (item_id/store_id), ds (date), y (units sold),
    sell_price, event_name_1, snap flag for the store's state.

Falls back to a synthetic generator (scripts/make_synthetic.py) when the M5
CSVs are absent, so the pipeline runs without Kaggle credentials.

Usage:
    python -m demandcast.data.prepare
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from demandcast.data.schemas import validate_sales
from demandcast.settings import get_config, resolve_path

logger = logging.getLogger(__name__)


def load_m5(raw_dir: Path) -> pd.DataFrame:
    sales = pd.read_csv(raw_dir / "sales_train_validation.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    cfg = get_config()["data"]

    sales = sales[sales["store_id"].isin(cfg["stores"])]
    # Top items by total units within the subset keeps the frame CPU-sized.
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    totals = sales.groupby("item_id")[day_cols].sum().sum(axis=1)
    keep_items = totals.nlargest(cfg["top_items"]).index
    sales = sales[sales["item_id"].isin(keep_items)]

    long = sales.melt(
        id_vars=["item_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="y",
    )
    calendar_cols = ["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]
    long = long.merge(calendar[calendar_cols], on="d", how="left")
    long["ds"] = pd.to_datetime(long["date"])
    long["snap"] = 0
    for state in ("CA", "TX", "WI"):
        mask = long["state_id"] == state
        long.loc[mask, "snap"] = long.loc[mask, f"snap_{state}"]

    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["unique_id"] = long["item_id"] + "/" + long["store_id"]
    out = long[["unique_id", "ds", "y", "sell_price", "event_name_1", "snap"]].sort_values(
        ["unique_id", "ds"]
    )
    return out.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = get_config()["data"]
    raw_dir = resolve_path(cfg["raw_dir"])

    if (raw_dir / "sales_train_validation.csv").exists():
        logger.info("Preparing M5 subset from %s", raw_dir)
        df = load_m5(raw_dir)
    elif (raw_dir / "synthetic_sales.parquet").exists():
        logger.info("M5 not found; using synthetic sales")
        df = pd.read_parquet(raw_dir / "synthetic_sales.parquet")
    else:
        raise FileNotFoundError(
            f"No raw data in {raw_dir}. Run scripts/make_synthetic.py or download M5 via "
            "`python -m demandcast.data.download`."
        )

    df = validate_sales(df)
    out_dir = resolve_path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sales.parquet"
    df.to_parquet(path, index=False)
    logger.info(
        "%d rows, %d series, %s -> %s -> %s",
        len(df),
        df["unique_id"].nunique(),
        df["ds"].min().date(),
        df["ds"].max().date(),
        path,
    )


if __name__ == "__main__":
    main()
