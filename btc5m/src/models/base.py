"""Forecaster interface. predict() uses each row's own features (as of that second)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    """Polymarket mid at the same second. Not fitted.

    When one side of the book is empty the missing bid counts as 0 and the missing
    ask as 1, so the seconds where the market is nearly decided are still scored.
    """

    name = "market"

    def __init__(self, clip_lo=0.005, clip_hi=0.995):
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    def predict(self, grid) -> pd.Series:
        return self._series(grid, grid["pm_mid_filled"].to_numpy(), self.clip_lo, self.clip_hi)


class LogisticForecaster(Forecaster):
    """L2 logistic regression on standardized features. Subclasses supply design()."""

    uses_spot = False

    def __init__(self, C=1.0, clip_lo=0.005, clip_hi=0.995):
        self.C = C
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.model = None
        self.feature_names = None

    def design(self, grid) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, grid_train):
        train = grid_train.loc[grid_train["tau"] >= 1]
        x = self.design(train)
        y = train["y"].to_numpy().astype(int)
        keep = np.isfinite(x.to_numpy()).all(axis=1)
        if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
            raise ValueError(f"{self.name} needs finite rows of both outcomes")
        self.feature_names = list(x.columns)
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=self.C, max_iter=2000)),
        ])
        self.model.fit(x.to_numpy()[keep], y[keep])
        return self

    def predict(self, grid):
        if self.model is None:
            raise RuntimeError(f"{self.name}.predict called before fit")
        values = self.design(grid).to_numpy()
        good = np.isfinite(values).all(axis=1)
        p = np.full(len(grid), np.nan)
        if good.any():
            classes = list(self.model.named_steps["logit"].classes_)
            p[good] = self.model.predict_proba(values[good])[:, classes.index(1)]
        return self._series(grid, p, self.clip_lo, self.clip_hi)

    def dp_dS(self, grid, dS=1.0):
        """One-step finite difference through predict. Flow features do not move with S."""
        if not self.uses_spot:
            return np.zeros(len(grid), dtype=float)
        p0 = self.predict(grid).to_numpy()
        bumped = grid.assign(S=grid["S"].to_numpy(dtype=float) + dS)
        slope = (self.predict(bumped).to_numpy() - p0) / dS
        slope[~np.isfinite(slope)] = 0.0
        return slope

    def coefficients(self):
        """Standardized-feature coefficients, for the written report."""
        logit = self.model.named_steps["logit"]
        return dict(zip(self.feature_names, logit.coef_.ravel().tolist()))
