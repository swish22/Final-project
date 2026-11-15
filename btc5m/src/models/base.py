"""Forecaster interface. predict() uses each row's own features (as of that second)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import clip_prob


class Forecaster:
    name = "base"

    def fit(self, grid_train):
        return self

    def predict(self, grid) -> pd.Series:
        raise NotImplementedError

    def dp_dS(self, grid) -> np.ndarray:
        """Sensitivity of P(Up) to the spot proxy. Zero when the model ignores spot."""
        return np.zeros(len(grid), dtype=float)

    def _series(self, grid, values, lo, hi) -> pd.Series:
        return pd.Series(clip_prob(values, lo, hi), index=grid.index, name=self.name)


class MarketMid(Forecaster):
    """Polymarket mid at the same second. Not fitted."""

    name = "market"

    def __init__(self, clip_lo=0.005, clip_hi=0.995):
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    def predict(self, grid) -> pd.Series:
        return self._series(grid, grid["pm_mid"].to_numpy(), self.clip_lo, self.clip_hi)
