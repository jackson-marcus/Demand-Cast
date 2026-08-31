"""Stage 2: turn the caller's stock picture into an inventory position.

Inventory position, not on-hand, is what an order-up-to policy compares
against: units already on a truck will arrive inside the protection window, so
ordering as if they did not exist double-orders them. This stage is the only
place the two are added together, and it refuses negatives outright -- a
negative on-hand usually means the caller is passing backorders in the wrong
sign, and silently accepting it inflates every order downstream.
"""

from __future__ import annotations

from typing import Any


def run(context: dict[str, Any]) -> dict[str, Any]:
    request = context["request"]
    if request.on_hand < 0:
        raise ValueError(f"on_hand must be >= 0, got {request.on_hand}")
    if request.in_transit < 0:
        raise ValueError(f"in_transit must be >= 0, got {request.in_transit}")

    context["position"] = {
        "on_hand": float(request.on_hand),
        "in_transit": float(request.in_transit),
        "inventory_position": float(request.on_hand + request.in_transit),
    }
    return context
