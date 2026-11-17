"""Brier score, log loss and calibration of lagged probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import HOUR_S, block_bootstrap, brier_score, log_loss


def lag_predictions(grid, model_cols, delta_s):
    """Probability published at t is the model output from the row at t - delta.

    Keeps the seconds where the market is still open (tau >= 1) and every model has
    a value, so all models are scored on exactly the same rows.
    """
    out = grid[["slug", "t", "split", "y", "tau", "start_unix"]].copy()
    shifted = grid.groupby("slug", sort=False)[list(model_cols)].shift(delta_s)
    for col in model_cols:
        out[col] = shifted[col].to_numpy()
    out = out.loc[(out["tau"] >= 1)].dropna(subset=list(model_cols))
    return out.reset_index(drop=True)


def tau_bucket(tau):
    tau = np.asarray(tau, dtype=float)
    labels = np.full(len(tau), "120+", dtype=object)
    labels[tau < 120] = "60-120"
    labels[tau < 60] = "30-60"
    labels[tau < 30] = "0-30"
    return labels


def score_predictions(lagged, models):
    rows = []
    frame = lagged.copy()
    frame["bucket"] = tau_bucket(frame["tau"].to_numpy())
    for split, by_split in frame.groupby("split", sort=False):
        slices = [("all", by_split)] + list(by_split.groupby("bucket", sort=False))
        for model in models:
            for bucket, group in slices:
                rows.append({
                    "split": split,
                    "model": model,
                    "tau_bucket": bucket,
                    "markets": int(group["slug"].nunique()),
                    "n": int(len(group)),
                    "brier": brier_score(group["y"], group[model]),
                    "log_loss": log_loss(group["y"], group[model]),
                })
    return pd.DataFrame(rows)


def brier_differences(lagged, models, baselines, split, n_boot=2000):
    """Brier(model) - Brier(baseline) on one split, with an hourly block-bootstrap interval.

    Negative means the model is more accurate than the baseline.
    """
    frame = lagged.loc[lagged["split"] == split]
    if frame.empty:
        return pd.DataFrame()
    y = frame["y"].to_numpy(dtype=float)
    per_market = pd.DataFrame({"slug": frame["slug"].to_numpy(), "n": 1.0,
                               "block": frame["start_unix"].to_numpy() // HOUR_S})
    for model in models:
        per_market[model] = (frame[model].to_numpy(dtype=float) - y) ** 2
    agg = per_market.groupby("slug", sort=False).agg({"n": "sum", "block": "first", **{m: "sum" for m in models}})
    rows = []
    for base in baselines:
        for model in models:
            if model == base:
                continue
            est, lo, hi, _ = block_bootstrap(agg[model] - agg[base], agg["n"], agg["block"], n_boot)
            rows.append({"split": split, "model": model, "baseline": base, "brier_diff": est,
                         "ci_lo": lo, "ci_hi": hi, "better": bool(hi < 0)})
    return pd.DataFrame(rows)


def calibration_bins(lagged, model, split, bins=10):
    frame = lagged.loc[lagged["split"] == split, ["y", model]]
    if frame.empty:
        return pd.DataFrame(columns=["p", "y", "n"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(frame[model].to_numpy(), edges) - 1, 0, bins - 1)
    table = frame.assign(bin=which).groupby("bin").agg(p=(model, "mean"), y=("y", "mean"), n=("y", "size"))
    return table.reset_index(drop=True)


def calibration_error(lagged, models, split, bins=10):
    """Expected calibration error: the average gap between predicted and realized frequency."""
    rows = []
    for model in models:
        table = calibration_bins(lagged, model, split, bins)
        if table.empty:
            continue
        ece = float((table["n"] * (table["p"] - table["y"]).abs()).sum() / table["n"].sum())
        rows.append({"split": split, "model": model, "ece": ece})
    return pd.DataFrame(rows)
