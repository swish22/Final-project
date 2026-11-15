"""G1: L2 logistic regression on Polymarket features only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import Forecaster

FEATURES = ["pm_mid", "last_trade", "mid_chg_30", "imbalance", "tau"]


class G1(Forecaster):
    name = "G1"

    def __init__(self, C=1.0, clip_lo=0.005, clip_hi=0.995):
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.C = C
        self.medians = None
        self.model = None

    def _matrix(self, grid, fit):
        x = grid[FEATURES].apply(pd_to_numeric)
        if fit:
            self.medians = x.median(numeric_only=True)
        x = x.fillna(self.medians)
        return x

    def fit(self, grid_train):
        x = self._matrix(grid_train, fit=True)
        y = grid_train["y"].to_numpy().astype(int)
        keep = np.isfinite(x.to_numpy()).all(axis=1)
        if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
            raise ValueError("G1 needs finite rows of both outcomes")
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=self.C, max_iter=2000, random_state=0)),
        ])
        self.feature_names = list(FEATURES)
        self.model.fit(x.to_numpy()[keep], y[keep])
        return self

    def predict(self, grid):
        if self.model is None:
            raise RuntimeError("G1.predict called before fit")
        x = self._matrix(grid, fit=False)
        values = x.to_numpy()
        good = np.isfinite(values).all(axis=1)
        p = np.full(len(grid), np.nan)
        if good.any():
            proba = self.model.predict_proba(values[good])
            p[good] = _class_probability(self.model.named_steps["logit"].classes_, proba)
        return self._series(grid, p, self.clip_lo, self.clip_hi)

    def coefficients(self):
        """Standardized-feature coefficients, for the written report."""
        logit = self.model.named_steps["logit"]
        return dict(zip(self.feature_names, logit.coef_.ravel().tolist()))


def pd_to_numeric(col):
    return col.astype(float)


def _class_probability(classes, proba):
    classes = list(classes)
    if 1 not in classes:
        return np.zeros(len(proba))
    return proba[:, classes.index(1)]
