"""Brier score and log loss of lagged probabilities, overall and by time left."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import brier_score, log_loss


def lag_predictions(grid, model_cols, delta_s):
    """Probability published at t is the model output from the row at t - delta."""
    out = grid[["slug", "t", "split", "y", "tau"]].copy()
    shifted = grid.groupby("slug", sort=False)[list(model_cols)].shift(delta_s)
    for col in model_cols:
        out[col] = shifted[col].to_numpy()
    return out


def tau_bucket(tau):
    tau = np.asarray(tau, dtype=float)
    labels = np.full(len(tau), "120+", dtype=object)
    labels[tau < 120] = "60-120"
    labels[tau < 60] = "30-60"
    labels[tau < 30] = "0-30"
    return labels


def score_predictions(lagged, models):
    rows = []
    frame = lagged.dropna(subset=["y"]).copy()
    frame["bucket"] = tau_bucket(frame["tau"].to_numpy())
    for split, by_split in frame.groupby("split", sort=False):
        slices = [("all", by_split)]
        for name, group in by_split.groupby("bucket", sort=False):
            slices.append((name, group))
        for model in models:
            for bucket, group in slices:
                usable = group.dropna(subset=[model])
                if usable.empty:
                    continue
                rows.append({
                    "split": split,
                    "model": model,
                    "tau_bucket": bucket,
                    "n": int(len(usable)),
                    "brier": brier_score(usable["y"], usable[model]),
                    "log_loss": log_loss(usable["y"], usable[model]),
                })
    return pd.DataFrame(rows)
