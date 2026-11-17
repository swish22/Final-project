"""Settlement PnL, where it came from, spread, drawdown and tail risk for a replay.

Every fill's profit at settlement splits into three parts that add up exactly:
    spread capture      distance from the mid just before the fill to our price
    30-second move      how the mid moved against (or for) us right after the fill
    held to settlement  what the position made from then until the market resolved
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import HOUR_S, block_bootstrap
from mm.replay import FEE_RATE, REBATE_SHARE


def cvar(pnls, alpha):
    """Mean of the worst ceil(alpha * n) market PnLs."""
    values = np.sort(np.asarray(pnls, dtype=float))
    if len(values) == 0:
        return float("nan")
    count = max(1, int(math.ceil(float(alpha) * len(values))))
    return float(values[:count].mean())


def wealth_curve(market_rows):
    """Cumulative settled PnL in time order, and the largest peak-to-trough loss.

    The trough inside a market uses its lowest mark-to-mid equity, so a loss that
    recovers before settlement still counts toward the drawdown.
    """
    rows = market_rows.sort_values("start_unix")
    pnl = rows["pnl"].to_numpy(dtype=float)
    low = rows["min_equity"].to_numpy(dtype=float)
    settled = np.cumsum(pnl)
    before = np.r_[0.0, settled[:-1]]
    peak = np.maximum.accumulate(np.r_[0.0, settled])[:-1]
    trough = np.minimum(before + low, settled)
    drawdown = float(np.max(peak - trough)) if len(pnl) else 0.0
    return rows["start_unix"].to_numpy(), settled, max(drawdown, 0.0)


def summarize(summaries, alpha=0.05, n_boot=2000):
    """One row per replay label. `summaries` has one row per market per label."""
    rows = []
    for label, g in summaries.groupby("label", sort=False):
        pnl = g["pnl"].to_numpy(dtype=float)
        blocks = g["start_unix"].to_numpy() // HOUR_S
        mean, lo, hi, se = block_bootstrap(pnl, np.ones(len(pnl)), blocks, n_boot)
        shares = float(g["shares"].sum())
        per_share = (lambda col: float(g[col].sum()) / shares * 100.0) if shares > 0 else (lambda col: float("nan"))
        _, _, drawdown = wealth_curve(g)
        both = float(g["quote_both_s"].sum())
        rows.append({
            "label": label,
            "markets": int(len(g)),
            "fills": int(g["n_fills"].sum()),
            "cross_fill_share": float(g["n_cross"].sum()) / max(1, int(g["n_fills"].sum())),
            "shares": shares,
            "pnl": float(pnl.sum()),
            "pnl_per_market": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "t_stat": mean / se if se and se > 0 else float("nan"),
            "win_rate": float((pnl > 0).mean()),
            "spread_capture_c": per_share("spread_capture"),
            "markout_5s_c": per_share("markout_5s"),
            "markout_30s_c": per_share("markout_30s"),
            "pnl_per_share_c": float(pnl.sum()) / shares * 100.0 if shares > 0 else float("nan"),
            "quoted_spread_c": float(g["spread_sum"].sum()) / both * 100.0 if both > 0 else float("nan"),
            "uptime": float(g["quote_any_s"].sum()) / float(g["seconds"].sum()),
            "mean_abs_inv": float(g["mean_abs_inv"].mean()),
            "max_drawdown": drawdown,
            "cvar": cvar(pnl, alpha),
            "worst_market": float(pnl.min()),
            "pnl_sd": float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan"),
            "rebate_est": float(g["fee_equiv"].sum()) * FEE_RATE * REBATE_SHARE,
        })
    return pd.DataFrame(rows)


def pnl_differences(summaries, pairs, n_boot=2000):
    """Mean per-market PnL of label a minus label b on the same markets, with an interval."""
    wide = summaries.pivot(index="slug", columns="label", values="pnl")
    blocks = summaries.drop_duplicates("slug").set_index("slug")["start_unix"].reindex(wide.index) // HOUR_S
    rows = []
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        diff = (wide[a] - wide[b]).to_numpy(dtype=float)
        mean, lo, hi, _ = block_bootstrap(diff, np.ones(len(diff)), blocks.to_numpy(), n_boot)
        rows.append({"a": a, "b": b, "pnl_diff_per_market": mean, "ci_lo": lo, "ci_hi": hi,
                     "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)
