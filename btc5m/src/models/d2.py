"""D2: L2 logistic recalibration of D1 plus short-horizon spot features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import logit
from models.base import Forecaster
from models.d1 import D1

# Drop the 0-30 bucket so the dummies are not collinear with the intercept.
BUCKETS = (("b30", 30, 60), ("b60", 60, 120), ("b120", 120, 10**9))


def _buckets(tau):
    tau = np.asarray(tau, dtype=float)
    data = {}
    for name, lo, hi in BUCKETS:
        data[name] = ((tau >= lo) & (tau < hi)).astype(float)
    return pd.DataFrame(data)


def design_matrix(grid, digital):
    """Columns: logit(D1), 10s Binance signed volume, venue gap, z, tau buckets."""
    p = digital.predict(grid).to_numpy()
    z = np.clip(digital.z_score(grid), -8.0, 8.0)
    frame = pd.DataFrame({
        "logit_d1": logit(p),
        "binance_signed_vol_10s": grid["binance_signed_vol_10s"].to_numpy(dtype=float),
        "cb_bn_gap": grid["cb_bn_gap"].to_numpy(dtype=float),
        "z": z,
    }, index=grid.index)
    buckets = _buckets(grid["tau"].to_numpy()).set_index(grid.index)
    return pd.concat([frame, buckets], axis=1)


class D2(Forecaster):
    name = "D2"

    def __init__(self, C=1.0, clip_lo=0.005, clip_hi=0.995):
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.C = C
        self.digital = D1(clip_lo, clip_hi)
        self.medians = None
        self.model = None
        self.feature_names = None

    def _matrix(self, grid, fit):
        x = design_matrix(grid, self.digital)
        if fit:
            self.medians = x.median(numeric_only=True)
        return x.fillna(self.medians)

    def fit(self, grid_train):
        x = self._matrix(grid_train, fit=True)
        y = grid_train["y"].to_numpy().astype(int)
        keep = np.isfinite(x.to_numpy()).all(axis=1)
        if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
            raise ValueError("D2 needs finite rows of both outcomes")
        self.feature_names = list(x.columns)
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=self.C, max_iter=2000, random_state=0)),
        ])
        self.model.fit(x.to_numpy()[keep], y[keep])
        return self

    def predict(self, grid):
        if self.model is None:
            raise RuntimeError("D2.predict called before fit")
        x = self._matrix(grid, fit=False)
        values = x.to_numpy()
        good = np.isfinite(values).all(axis=1)
        p = np.full(len(grid), np.nan)
        if good.any():
            logit_model = self.model.named_steps["logit"]
            proba = self.model.predict_proba(values[good])
            classes = list(logit_model.classes_)
            p[good] = proba[:, classes.index(1)] if 1 in classes else 0.0
        return self._series(grid, p, self.clip_lo, self.clip_hi)

    def dp_dS(self, grid, dS=1.0):
        """One-step finite difference through predict. Spot features move with S; flow does not."""
        p0 = self.predict(grid).to_numpy()
        bumped = grid.copy()
        bumped["S"] = grid["S"].to_numpy(dtype=float) + dS
        p1 = self.predict(bumped).to_numpy()
        slope = (p1 - p0) / dS
        slope[~np.isfinite(slope)] = 0.0
        return slope

    def coefficients(self):
        logit_model = self.model.named_steps["logit"]
        return dict(zip(self.feature_names, logit_model.coef_.ravel().tolist()))
