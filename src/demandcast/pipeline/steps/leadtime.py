"""Stage 4: aggregate the daily fan across the protection window.

This is the stage that actually decides how much inventory the plan carries,
and it is the one planners get wrong by hand. The intuitive move is to add up
the daily P90s -- "cover the bad case every day" -- but that is the quantile of
the sum only if every day's demand is perfectly correlated with every other's.
Adding variances instead assumes daily demand is independent, so the spread
grows with ``sqrt(n)`` rather than ``n``.

Both are computed here. The comonotone figure is not used for the order; it is
carried through to the response so the caller can see what the shortcut would
have cost them. On the 28-day holdout the shortcut holds 79% more inventory
for 1.9 points of fill rate (``scripts/leadtime_study.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def run(context: dict[str, Any]) -> dict[str, Any]:
    fan = context["fan"]
    sigma = context["spread"]["sigma_daily"]

    expected = float(np.sum(fan["q50"]))
    independent = float(np.sqrt(np.sum(np.square(sigma))))
    comonotone = float(np.sum(sigma))

    context["leadtime"] = {
        "days": fan["days"],
        "expected_demand": expected,
        "sigma": independent,
        "comonotone_sigma": comonotone,
    }
    return context
