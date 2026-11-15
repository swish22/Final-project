"""G0: the training base rate, one label per market."""
from __future__ import annotations

import numpy as np

from models.base import Forecaster


class G0(Forecaster):
    name = "G0"

    def __init__(self, clip_lo=0.005, clip_hi=0.995):
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.p = None

    def fit(self, grid_train):
        labels = grid_train.groupby("slug", sort=False)["y"].first()
        if labels.empty:
            raise ValueError("G0 has no training markets")
        self.p = float(labels.mean())
        return self

    def predict(self, grid):
        if self.p is None:
            raise RuntimeError("G0.predict called before fit")
        return self._series(grid, np.full(len(grid), self.p), self.clip_lo, self.clip_hi)
