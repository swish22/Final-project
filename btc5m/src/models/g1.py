"""G1: L2 logistic regression on Polymarket features only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import logit, signed_sqrt
from models.base import LogisticForecaster


def market_features(grid):
    """Polymarket-only features. Prices enter as log-odds so the model can reproduce the mid."""
    mid = grid["pm_mid_filled"].astype(float)
    last = grid["last_trade"].astype(float).fillna(mid)
    return pd.DataFrame({
        "logit_mid": np.where(mid.notna(), logit(mid.clip(0.005, 0.995)), np.nan),
        "logit_last": np.where(last.notna(), logit(last.clip(0.005, 0.995)), np.nan),
        "mid_chg_30": grid["mid_chg_30"].astype(float).fillna(0.0),
        "imbalance": grid["imbalance"].astype(float).fillna(0.0),
        "pm_flow_30": signed_sqrt(grid["pm_flow_30"].astype(float).fillna(0.0)),
        "tau": grid["tau"].astype(float) / 300.0,
    }, index=grid.index)


class G1(LogisticForecaster):
    name = "G1"

    def design(self, grid):
        return market_features(grid)
