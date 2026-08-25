"""Pandera schema for the canonical long sales frame."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column

SALES = pa.DataFrameSchema(
    columns={
        "unique_id": Column(str),
        "ds": Column("datetime64[ns]"),
        "y": Column(float, Check.ge(0), coerce=True),
        "sell_price": Column(float, nullable=True, coerce=True),
        "snap": Column(int, Check.isin([0, 1]), coerce=True),
    },
    strict=False,
    unique=["unique_id", "ds"],
)


def validate_sales(df):
    return SALES.validate(df, lazy=True)
