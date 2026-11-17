"""Shared helpers for the BTC 5-minute backtest."""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

TICK = 0.01
HOUR_S = 3600


def be_nice():
    """Drop this process to below-normal priority so a long run leaves the machine usable."""
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception:
        pass


def clip_prob(p, lo, hi):
    return np.clip(np.asarray(p, dtype=float), lo, hi)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def signed_sqrt(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.sqrt(np.abs(x))


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


def mills(x):
    """One price in dollars -> integer tenths of a cent, or -1 when it is missing."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return -1
    if not math.isfinite(v) or v <= 0:
        return -1
    return int(round(v * 1000.0))


def mills_array(values):
    """Prices in dollars -> integer tenths of a cent, with -1 where the level is missing.

    Polymarket's tick is 0.001 near 0 and 1, so cents are not fine enough for book prices.
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    out = np.full(v.shape, -1, dtype=np.int64)
    out[ok] = np.rint(v[ok] * 1000.0).astype(np.int64)
    return out


def assign_split(day_idx, split_cfg):
    """split_cfg maps a split name to an inclusive [first_day, last_day] range of day indexes."""
    days = np.asarray(day_idx)
    out = np.full(len(days), None, dtype=object)
    for name, (lo, hi) in split_cfg.items():
        out[(days >= int(lo)) & (days <= int(hi))] = name
    return out


def brier_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def block_bootstrap(numer, denom, blocks, n_boot=2000, seed=0, level=0.95):
    """Ratio sum(numer) / sum(denom) with a confidence interval from resampling whole blocks.

    Markets in the same block (one hour) are kept together, so correlation between
    neighbouring markets does not shrink the interval. Returns (estimate, lo, hi, se).
    """
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    if len(numer) == 0 or denom.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    frame = pd.DataFrame({"n": numer, "d": denom, "b": np.asarray(blocks)})
    by = frame.groupby("b", sort=False)[["n", "d"]].sum().to_numpy()
    estimate = float(by[:, 0].sum() / by[:, 1].sum())
    if len(by) < 2:
        return estimate, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(by), size=(n_boot, len(by)))
    num = by[draws, 0].sum(axis=1)
    den = by[draws, 1].sum(axis=1)
    stats = num[den > 0] / den[den > 0]
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [tail, 1.0 - tail])
    return estimate, float(lo), float(hi), float(stats.std(ddof=1))
