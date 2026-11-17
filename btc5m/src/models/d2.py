"""D2: L2 logistic recalibration of D1 plus short-horizon spot features.

D3 adds the Polymarket features to D2, to measure what the BTC model knows that the
market price does not already contain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import logit, signed_sqrt
from models.base import LogisticForecaster
from models.d1 import D1
from models.g1 import market_features

# Drop the 0-30 bucket so the dummies are not collinear with the intercept.
BUCKETS = (("b30", 30, 60), ("b60", 60, 120), ("b120", 120, 10**9))


def _buckets(tau, index):
    tau = np.asarray(tau, dtype=float)
    data = {name: ((tau >= lo) & (tau < hi)).astype(float) for name, lo, hi in BUCKETS}
    return pd.DataFrame(data, index=index)


def spot_features(grid, digital):
    """logit(D1), 10s Binance signed volume, venue gap in sigma units, z, tau buckets."""
    p = digital.predict(grid).to_numpy()
    z = np.clip(digital.z_score(grid), -8.0, 8.0)
    tau = grid["tau"].to_numpy(dtype=float)
    scale = grid["S"].to_numpy(dtype=float) * grid["sigma"].to_numpy(dtype=float) * np.sqrt(np.clip(tau, 1.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_z = grid["gap_chg"].to_numpy(dtype=float) / scale
    gap_z = np.where(np.isfinite(gap_z), np.clip(gap_z, -8.0, 8.0), 0.0)
    frame = pd.DataFrame({
        "logit_d1": np.where(np.isfinite(p), logit(p), np.nan),
        "z": z,
        "binance_flow_10s": signed_sqrt(grid["binance_signed_vol_10s"].astype(float).fillna(0.0)),
        "gap_z": gap_z,
    }, index=grid.index)
    return pd.concat([frame, _buckets(tau, grid.index)], axis=1)


class D2(LogisticForecaster):
    name = "D2"
    uses_spot = True

    def __init__(self, C=1.0, clip_lo=0.005, clip_hi=0.995):
        super().__init__(C, clip_lo, clip_hi)
        self.digital = D1(clip_lo, clip_hi)

    def design(self, grid):
        return spot_features(grid, self.digital)


class D3(D2):
    name = "D3"

    def design(self, grid):
        return pd.concat([spot_features(grid, self.digital), market_features(grid).drop(columns="tau")], axis=1)
