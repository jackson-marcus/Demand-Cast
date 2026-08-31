"""Stage 5: the order-up-to level and the quantity to order today.

Base-stock policy under daily review::

    S = expected lead-time demand + z(service_level) * sigma_lead_time
    order = max(0, S - inventory_position)

``max(0, ...)`` is not cosmetic: when the position already exceeds S the
correct action is to order nothing, and a policy that returned a negative
number would read as a return-to-vendor instruction it has no business
issuing.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any


def run(context: dict[str, Any]) -> dict[str, Any]:
    request = context["request"]
    leadtime = context["leadtime"]
    position = context["position"]["inventory_position"]

    z = NormalDist().inv_cdf(request.service_level)
    safety = z * leadtime["sigma"]
    order_up_to = leadtime["expected_demand"] + safety
    comonotone_order_up_to = leadtime["expected_demand"] + z * leadtime["comonotone_sigma"]

    context["order"] = {
        "service_level": request.service_level,
        "z": z,
        "order_up_to": order_up_to,
        "safety_stock": safety,
        "order_quantity": max(0.0, order_up_to - position),
        "comonotone_order_up_to": comonotone_order_up_to,
        "comonotone_extra_units": comonotone_order_up_to - order_up_to,
    }
    return context
