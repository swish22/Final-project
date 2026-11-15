"""Shared helpers for the BTC 5-minute backtest."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TICK = 0.01


def clip_prob(p, lo, hi):
    return np.clip(np.asarray(p, dtype=float), lo, hi)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def to_tick(x):
    """Nearest 1-cent price, or None when the level is missing."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0:
        return None
    return round(v * 100.0) / 100.0


def cents(x):
    tick = to_tick(x)
    if tick is None:
        return None
    return int(round(tick * 100.0))


def lag_frame(df, cols, delta_s):
    """Shift columns forward by delta_s seconds within each market.

    Row t then holds the values from t - delta_s. The frame must contain
    one row per second, sorted by slug and t.
    """
    out = df.copy()
    if delta_s <= 0:
        return out
    shifted = out.groupby("slug", sort=False)[cols].shift(delta_s)
    out[cols] = shifted
    return out


def utc_day(start_unix):
    return pd.to_datetime(start_unix, unit="s", utc=True).strftime("%Y-%m-%d")


def assign_split(days, split_cfg):
    lookup = {}
    for name, values in split_cfg.items():
        for day in values:
            lookup[day] = name
    return pd.Series(days).map(lookup).to_numpy()


def brier_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
