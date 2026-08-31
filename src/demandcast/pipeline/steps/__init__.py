"""Stages of the replenishment plan, keyed by the name used in the DAG."""

from collections.abc import Callable
from typing import Any

from . import audit, calibrate, fan, leadtime, policy, position

StageRunner = Callable[[dict[str, Any]], dict[str, Any]]

STAGE_RUNNERS: dict[str, StageRunner] = {
    "fan": fan.run,
    "position": position.run,
    "calibrate": calibrate.run,
    "leadtime": leadtime.run,
    "policy": policy.run,
    "audit": audit.run,
}
