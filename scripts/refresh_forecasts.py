"""Batch job: fit the champion on all data and persist forecasts for serving.

Usage:
    uv run python scripts/refresh_forecasts.py [--model lgbm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandcast.models.backtest import FORECASTERS, load_sales
from demandcast.settings import get_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lgbm", choices=sorted(FORECASTERS))
    args = parser.parse_args()

    cfg = get_config()
    df = load_sales()
    pred = FORECASTERS[args.model](df, cfg["forecast"]["horizon"], cfg["forecast"]["quantiles"])
    pred["model"] = args.model

    out_dir = resolve_path(cfg["data"]["forecasts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "latest.parquet"
    pred.to_parquet(out, index=False)
    print(f"Wrote {len(pred):,} forecast rows for {pred['unique_id'].nunique()} series -> {out}")


if __name__ == "__main__":
    main()
