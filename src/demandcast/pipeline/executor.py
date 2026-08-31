"""Run the replenishment plan for one series.

The planner holds the two frames every stage needs (the stored forecast fan and
the realised sales panel) and walks :data:`demandcast.pipeline.dag.STAGES` in
dependency order, handing each stage a shared context dict. After each stage it
checks that the key that stage declared it produces is actually there, so a
stage that returns without doing its job is caught where it happened rather
than three stages later as a ``KeyError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from demandcast.pipeline.dag import STAGES_BY_NAME, topological_order
from demandcast.pipeline.steps import STAGE_RUNNERS
from demandcast.settings import get_config

logger = logging.getLogger(__name__)

#: Order quantities below this many units are not worth a purchase order.
MIN_ORDER_UNITS = 0.5


@dataclass(frozen=True)
class ReplenishmentRequest:
    """Everything the plan needs that is not already in the data."""

    unique_id: str
    on_hand: float = 0.0
    in_transit: float = 0.0
    lead_time: int = 7
    service_level: float = 0.9
    audit_days: int = 28
    sigma_history_days: int = 63
    floor_sigma: bool = True

    @classmethod
    def from_config(cls, unique_id: str, **overrides: Any) -> ReplenishmentRequest:
        """Build a request, filling anything left as ``None`` from config.yaml."""
        cfg = get_config()["replenishment"]
        defaults = {
            "lead_time": cfg["lead_time_days"],
            "service_level": cfg["service_level"],
            "audit_days": cfg["audit_days"],
            "sigma_history_days": cfg["sigma_history_days"],
        }
        fields = {k: v for k, v in overrides.items() if v is not None}
        return cls(unique_id=unique_id, **{**defaults, **fields})

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError(f"lead_time must be >= 0, got {self.lead_time}")
        if not 0.5 <= self.service_level < 1.0:
            raise ValueError(f"service_level must be in [0.5, 1.0), got {self.service_level}")
        if self.audit_days < 1:
            raise ValueError(f"audit_days must be >= 1, got {self.audit_days}")


class StageContractError(RuntimeError):
    """A stage ran but did not produce the context key it declared."""


class ReplenishmentPlanner:
    """Runs the staged plan against a forecast fan and a sales history."""

    def __init__(self, forecasts: pd.DataFrame, history: pd.DataFrame) -> None:
        self.forecasts = forecasts
        self.history = history

    def run(self, request: ReplenishmentRequest) -> dict[str, Any]:
        context: dict[str, Any] = {
            "request": request,
            "forecasts": self.forecasts,
            "history": self.history,
            "stages": [],
        }
        for name in topological_order():
            context = STAGE_RUNNERS[name](context)
            produced = STAGES_BY_NAME[name].produces
            if produced not in context:
                raise StageContractError(f"stage {name!r} did not produce {produced!r}")
            context["stages"].append(name)
        context["status"] = decide_status(context)
        logger.info(
            "replenish %s: %s, order %.1f units (audited fill %.3f)",
            request.unique_id,
            context["status"],
            context["order"]["order_quantity"],
            context["audit"]["fill_rate"],
        )
        return context


def decide_status(context: dict[str, Any]) -> str:
    """``review`` when the audit says the level would have missed the target.

    ``hold`` when the position already covers the level, otherwise ``order``.
    The review flag wins: a level that has just failed against real demand is
    worth a human look even if there is nothing to order today.
    """
    if not context["audit"]["meets_target"]:
        return "review"
    if context["order"]["order_quantity"] < MIN_ORDER_UNITS:
        return "hold"
    return "order"
