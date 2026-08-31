"""Typer CLI for the replenishment planner."""

from __future__ import annotations

import pandas as pd
import typer

from demandcast.models.backtest import load_sales
from demandcast.pipeline.dag import STAGES_BY_NAME, topological_order
from demandcast.pipeline.executor import ReplenishmentPlanner, ReplenishmentRequest
from demandcast.settings import get_config, resolve_path

app = typer.Typer(help="Plan replenishment from the stored probabilistic forecast.")


@app.command("plan")
def plan() -> None:
    """Print the replenishment stages in dependency order."""
    for name in topological_order():
        stage = STAGES_BY_NAME[name]
        deps = ", ".join(stage.requires) or "-"
        typer.echo(f"{name:<10} <- {deps:<22} {stage.summary}")


@app.command("replenish")
def replenish(
    unique_id: str = typer.Argument(..., help="series id, e.g. ITEM_001/ST_1"),
    on_hand: float = typer.Option(0.0, help="units physically in stock"),
    in_transit: float = typer.Option(0.0, help="units already on order"),
    lead_time: int = typer.Option(7, help="supplier lead time in days"),
    service_level: float = typer.Option(0.9, help="target fill rate"),
) -> None:
    """Recommend an order quantity for one series."""
    forecasts_path = resolve_path(get_config()["data"]["forecasts_dir"]) / "latest.parquet"
    if not forecasts_path.exists():
        typer.echo(f"No forecasts at {forecasts_path}; run scripts/refresh_forecasts.py", err=True)
        raise typer.Exit(code=1)

    planner = ReplenishmentPlanner(pd.read_parquet(forecasts_path), load_sales())
    result = planner.run(
        ReplenishmentRequest(
            unique_id=unique_id,
            on_hand=on_hand,
            in_transit=in_transit,
            lead_time=lead_time,
            service_level=service_level,
        )
    )
    order, audit = result["order"], result["audit"]
    typer.echo(f"{unique_id}: {result['status'].upper()} {order['order_quantity']:.1f} units")
    typer.echo(
        f"  order-up-to {order['order_up_to']:.1f} "
        f"(demand {result['leadtime']['expected_demand']:.1f} "
        f"+ safety {order['safety_stock']:.1f}) over {result['leadtime']['days']} days"
    )
    typer.echo(
        f"  audited over {audit['window_days']}d of actuals: fill {audit['fill_rate']:.1%} "
        f"vs target {audit['target_fill_rate']:.0%}, {audit['stockout_days']} stockout days"
    )


def main() -> None:
    app()
