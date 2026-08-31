"""Stage graph for the replenishment plan.

Turning a stored quantile fan into an order quantity is not one calculation,
it is a handful of them with real dependencies: the forecast band has to be
repaired and given a spread before it can be aggregated over a lead time, and
the aggregate is useless until it meets the caller's on-hand position. Two of
the stages have no inputs at all (``fan`` reads parquet, ``position`` reads the
request), so the graph genuinely branches and re-joins rather than being a
straight line dressed up as a DAG.

Each stage also declares the context key it is responsible for producing;
:mod:`demandcast.pipeline.executor` enforces that, so a stage that silently
does nothing fails loudly instead of leaving a hole for a later stage to trip
over.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One stage of the replenishment plan."""

    name: str
    requires: tuple[str, ...]
    produces: str
    summary: str


STAGES: tuple[Stage, ...] = (
    Stage("fan", (), "fan", "read the stored quantile forecast over the protection window"),
    Stage("position", (), "position", "normalise on-hand + in-transit into an inventory position"),
    Stage("calibrate", ("fan",), "spread", "repair crossed quantiles, floor the implied sigma"),
    Stage("leadtime", ("calibrate",), "leadtime", "aggregate the daily fan over the lead time"),
    Stage("policy", ("leadtime", "position"), "order", "order-up-to level and order quantity"),
    Stage("audit", ("policy",), "audit", "replay that level against recently realised demand"),
)

STAGES_BY_NAME: dict[str, Stage] = {stage.name: stage for stage in STAGES}


def topological_order(stages: Sequence[Stage] = STAGES) -> list[str]:
    """Stage names in an order that respects every dependency.

    Ties are broken by declaration order, which keeps the sequence stable
    across runs (the API reports it back, so it must not wobble).
    """
    position = {stage.name: i for i, stage in enumerate(stages)}
    remaining = {stage.name: set(stage.requires) for stage in stages}
    for name, deps in remaining.items():
        unknown = deps - remaining.keys()
        if unknown:
            raise ValueError(f"stage {name!r} depends on unknown stage(s) {sorted(unknown)}")

    ordered: list[str] = []
    while remaining:
        ready = [name for name, deps in remaining.items() if not deps]
        if not ready:
            raise RuntimeError(f"cycle in replenishment plan among {sorted(remaining)}")
        name = min(ready, key=position.__getitem__)
        ordered.append(name)
        del remaining[name]
        for deps in remaining.values():
            deps.discard(name)
    return ordered
