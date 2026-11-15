"""Settlement PnL, markout, drawdown, and CVaR for a replay."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def cvar(pnls, alpha):
    """Mean of the worst ceil(alpha * n) market PnLs. With 5 markets and alpha 0.2 this is the worst market."""
    values = np.sort(np.asarray(pnls, dtype=float))
    if len(values) == 0:
        return float("nan")
    count = max(1, int(math.ceil(float(alpha) * len(values))))
    return float(values[:count].mean())


def max_drawdown(wealth):
    wealth = np.asarray(wealth, dtype=float)
    if len(wealth) == 0:
        return 0.0
    peak = np.maximum.accumulate(wealth)
    return float(np.max(peak - wealth))


def attach_markout(fills, grid):
    if fills is None or len(fills) == 0:
        return fills
    out = fills.copy()
    mids = grid[["slug", "t", "pm_mid"]].drop_duplicates(["slug", "t"])
    lookup = {(row.slug, int(row.t)): row.pm_mid for row in mids.itertuples(index=False)}
    sign = np.where(out["side"].to_numpy() == "buy", 1.0, -1.0)
    for horizon, column in ((5, "markout_5s"), (30, "markout_30s")):
        future = [
            lookup.get((slug, int(t) + horizon), np.nan)
            for slug, t in zip(out["slug"], out["t"])
        ]
        out[column] = sign * (np.asarray(future, dtype=float) - out["price"].to_numpy(dtype=float))
    return out


def cumulative_wealth(paths):
    """Mark-to-mid wealth concatenated across markets, settled at each market end."""
    if paths is None or len(paths) == 0:
        return np.array([0.0]), np.array([0.0])
    wealth = [0.0]
    clock = [0.0]
    cursor = 0.0
    running = 0.0
    order = (
        paths.loc[~paths["settled"].astype(bool)]
        .groupby("slug", sort=False)["t"].min()
        .sort_values()
    )
    for slug in order.index:
        market = paths.loc[paths["slug"] == slug]
        for equity in market.loc[~market["settled"].astype(bool), "equity"]:
            cursor += 1
            clock.append(cursor)
            wealth.append(running + float(equity))
        settled = market.loc[market["settled"].astype(bool), "equity"]
        settle = float(settled.iloc[-1]) if len(settled) else 0.0
        running += settle
        cursor += 1
        clock.append(cursor)
        wealth.append(running)
    return np.asarray(clock), np.asarray(wealth)


def summarize(fills, paths, alpha=0.2):
    """One row per model, fill rule, and split, plus a per-market PnL table."""
    metric_rows = []
    market_rows = []
    if paths is None or len(paths) == 0:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["model", "fill_rule", "split"]
    for key, group in paths.groupby(keys, sort=False):
        model, rule, split = key
        settled = group.loc[group["settled"].astype(bool)]
        pnls = settled["equity"].to_numpy(dtype=float)
        _, wealth = cumulative_wealth(group)
        if fills is None or len(fills) == 0:
            ours = settled.iloc[0:0]
        else:
            mask = (
                (fills["model"] == model)
                & (fills["fill_rule"] == rule)
                & (fills["split"] == split)
            )
            ours = fills.loc[mask]
        metric_rows.append({
            "split": split,
            "model": model,
            "fill_rule": rule,
            "n_markets": int(len(settled)),
            "n_fills": int(len(ours)),
            "shares": float(ours["size"].sum()) if len(ours) else 0.0,
            "pnl": float(pnls.sum()) if len(pnls) else 0.0,
            "mean_market_pnl": float(pnls.mean()) if len(pnls) else 0.0,
            "max_drawdown": max_drawdown(wealth),
            "cvar": cvar(pnls, alpha),
            "markout_5s": _weighted(ours, "markout_5s"),
            "markout_30s": _weighted(ours, "markout_30s"),
        })
        for row in settled.itertuples(index=False):
            market_rows.append({
                "split": split,
                "model": model,
                "fill_rule": rule,
                "slug": row.slug,
                "pnl": row.equity,
            })
    metrics = pd.DataFrame(metric_rows)
    markets = pd.DataFrame(market_rows)
    if len(metrics):
        metrics = metrics.sort_values(["split", "fill_rule", "model"]).reset_index(drop=True)
    return metrics, markets


def _weighted(fills, column):
    if fills is None or len(fills) == 0 or column not in fills.columns:
        return float("nan")
    usable = fills.dropna(subset=[column])
    if usable.empty or float(usable["size"].sum()) <= 0:
        return float("nan")
    return float(np.average(usable[column], weights=usable["size"]))
