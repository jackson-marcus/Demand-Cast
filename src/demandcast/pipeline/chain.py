"""Transformer Chain Architecture - Composable Pipeline Chain.

Chains multiple TransformerStep objects sequentially, guaranteeing non-leakage
and consistent multi-step feature generation across training and inference.
"""

from __future__ import annotations

import pandas as pd

from demandcast.pipeline.base import TransformerStep
from demandcast.pipeline.transformers import (
    CalendarFeatureTransformer,
    LagFeatureTransformer,
    RollingStatTransformer,
)
from demandcast.settings import get_config


class TransformerChain:
    """Sequential chain of TransformerStep objects."""

    def __init__(self, steps: list[tuple[str, TransformerStep]] | None = None) -> None:
        self.steps = steps or []

    def add_step(self, name: str, step: TransformerStep) -> TransformerChain:
        self.steps.append((name, step))
        return self

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> TransformerChain:
        curr = df
        for _, step in self.steps:
            curr = step.fit_transform(curr, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        curr = df
        for _, step in self.steps:
            curr = step.transform(curr)
        return curr

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


def build_default_feature_chain() -> TransformerChain:
    """Construct the standard feature engineering transformer chain."""
    cfg = get_config()["features"]
    chain = TransformerChain()
    chain.add_step("calendar", CalendarFeatureTransformer())
    chain.add_step("lags", LagFeatureTransformer(lags=cfg["lags"]))
    chain.add_step("rolling", RollingStatTransformer(windows=cfg["rolling_windows"]))
    return chain
