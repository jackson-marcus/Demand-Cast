"""Transformer Chain Architecture - Base Step Protocol.

Defines the scikit-learn compatible fit/transform contract for feature engineering
and forecasting transformations.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class TransformerStep(Protocol):
    """Protocol for a composable transformation step."""

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> TransformerStep:
        """Fit internal parameters on training dataset."""
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset and return transformed DataFrame."""
        ...

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in a single call."""
        ...
