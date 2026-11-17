"""Does the BTC model move before the Polymarket price does?

Correlation between the one-second change in the Polymarket mid and the one-second
change in a model's probability k seconds earlier, both in log-odds. A large value at
k = 1 and a small one at k = -1 means BTC leads Polymarket by about a second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import logit


def lead_lag(frame, model, lags=range(-5, 6), tau_range=(20, 280)):
    f = frame.loc[frame["tau"].between(*tau_range), ["slug", "t", "pm_mid_filled", f"p_{model}"]].copy()
    f["mid"] = logit(f["pm_mid_filled"].clip(0.01, 0.99))
    f["model"] = logit(f[f"p_{model}"].clip(0.01, 0.99))
    by = f.groupby("slug", sort=False)
    f["d_mid"] = by["mid"].diff()
    f["d_model"] = by["model"].diff()
    by = f.groupby("slug", sort=False)
    rows = []
    for k in lags:
        earlier = by["d_model"].shift(k)
        ok = earlier.notna() & f["d_mid"].notna()
        corr = float(np.corrcoef(earlier[ok], f.loc[ok, "d_mid"])[0, 1]) if ok.sum() > 2 else float("nan")
        rows.append({"lag_s": int(k), "corr": corr, "n": int(ok.sum())})
    return pd.DataFrame(rows)
