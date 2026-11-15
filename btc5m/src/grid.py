"""Causal one-second grid built from the converter's Parquet tables.

Each row at second t stores the last observation with timestamp <= t.
Latency is applied later, by reading the row at t - delta, so the grid
itself does not depend on the latency setting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UP_LEVELS = ("up_bid1", "up_bid2", "up_ask1", "up_ask2")


def load_tables(parquet_dir):
    root = Path(parquet_dir)
    frames = {}
    for name in ("markets", "book", "trades", "btc_1s"):
        files = sorted((root / name).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files in {root / name}")
        frames[name] = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return frames["markets"], frames["book"], frames["trades"], frames["btc_1s"]


def ewma_sigma(prices, half_life_s):
    """sqrt of an EWMA of 1-second log-return variance.

    The recursion only uses returns at or before each second. Before the
    first return, sigma is NaN. Units are per sqrt(second).
    """
    s = np.asarray(prices, dtype=float)
    n = len(s)
    returns = np.full(n, np.nan)
    if n >= 2:
        prev, cur = s[:-1], s[1:]
        ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0) & (cur > 0)
        returns[1:][ok] = np.log(cur[ok] / prev[ok])
    lam = 0.5 ** (1.0 / float(half_life_s))
    var = np.full(n, np.nan)
    prev_var = None
    for i, r in enumerate(returns):
        if np.isfinite(r):
            if prev_var is None:
                prev_var = float(r * r)
            else:
                prev_var = lam * prev_var + (1.0 - lam) * float(r * r)
        if prev_var is not None:
            var[i] = prev_var
    return np.sqrt(var)


def _proxy(binance, coinbase):
    mids = np.vstack([np.asarray(binance, dtype=float), np.asarray(coinbase, dtype=float)])
    with np.errstate(all="ignore"):
        return np.nanmean(mids, axis=0)


def _clean_book(book):
    b = book.sort_values("ts").drop_duplicates("ts", keep="last").copy()
    b["ts"] = b["ts"].astype(float)
    for level in UP_LEVELS:
        price = b[level].astype(float)
        missing = ~np.isfinite(price) | (price < 0)
        b[level] = price.mask(missing)
        size_col = level + "_size"
        b[size_col] = b[size_col].astype(float).mask(missing)
    return b


def _spot_frame(btc, end):
    path = btc.sort_values("sec").drop_duplicates("sec", keep="last")
    sec0 = int(path["sec"].min())
    secs = np.arange(sec0, int(end) + 1, dtype=np.int64)
    path = path.set_index("sec").reindex(secs)
    path["binance_mid"] = path["binance_mid"].ffill()
    path["coinbase_mid"] = path["coinbase_mid"].ffill()
    path["binance_signed_vol"] = path["binance_signed_vol"].fillna(0.0)
    path["S"] = _proxy(path["binance_mid"].to_numpy(), path["coinbase_mid"].to_numpy())
    path.index.name = "sec"
    return path.reset_index()


def build_market_grid(market, btc, book, trades, half_life_s):
    """One market -> (grid rows from start to end, spot path used for sigma)."""
    slug = market["slug"]
    start, end = int(market["start_unix"]), int(market["end_unix"])
    spot = _spot_frame(btc, end)
    spot["sigma"] = ewma_sigma(spot["S"].to_numpy(), half_life_s)
    spot["binance_signed_vol_10s"] = spot["binance_signed_vol"].rolling(10, min_periods=1).sum()
    spot["slug"] = slug

    # Extend 30 seconds before the open so the 30-second mid change is causal at t = start.
    secs = np.arange(start - 30, end + 1, dtype=np.int64)
    frame = pd.DataFrame({"t": secs})
    spot_on = spot[["sec", "binance_mid", "coinbase_mid", "S", "sigma", "binance_signed_vol_10s"]].copy()
    frame = frame.merge(spot_on, left_on="t", right_on="sec", how="left")
    frame["t_key"] = frame["t"].astype(float)

    b = _clean_book(book)
    keep = ["ts", "up_bid1", "up_bid1_size", "up_ask1", "up_ask1_size"]
    merged = pd.merge_asof(
        frame.sort_values("t_key"),
        b[keep].sort_values("ts"),
        left_on="t_key",
        right_on="ts",
        direction="backward",
    )
    both = merged["up_bid1"].notna() & merged["up_ask1"].notna()
    merged["pm_mid"] = np.where(both, (merged["up_bid1"] + merged["up_ask1"]) / 2.0, np.nan)
    den = merged["up_bid1_size"] + merged["up_ask1_size"]
    merged["imbalance"] = np.where(both & (den > 0),
                                   (merged["up_bid1_size"] - merged["up_ask1_size"]) / den,
                                   np.nan)
    merged["mid_chg_30"] = merged["pm_mid"] - merged["pm_mid"].shift(30)

    if trades is None or len(trades) == 0:
        merged["last_trade"] = np.nan
    else:
        tr = trades.sort_values("ts").drop_duplicates("ts", keep="last")
        tr = tr[["ts", "up_price"]].rename(columns={"up_price": "last_trade"})
        tr["ts"] = tr["ts"].astype(float)
        merged = pd.merge_asof(
            merged.drop(columns=["ts"], errors="ignore").sort_values("t_key"),
            tr.sort_values("ts"),
            left_on="t_key",
            right_on="ts",
            direction="backward",
        )

    grid = merged.loc[(merged["t"] >= start) & (merged["t"] <= end)].copy()
    grid["t"] = grid["t"].astype(np.int64)
    s_open_rows = spot.loc[spot["sec"] == start, "S"]
    s_open = float(s_open_rows.iloc[0]) if len(s_open_rows) else float("nan")
    k = float(market["K"])
    grid["slug"] = slug
    grid["tau"] = end - grid["t"]
    grid["K"] = k
    grid["S_open"] = s_open
    grid["b"] = k - s_open
    grid["winner"] = market["winner"]
    grid["y"] = 1 if market["winner"] == "up" else 0
    grid["cb_bn_gap"] = grid["coinbase_mid"] - grid["binance_mid"]
    grid["start_unix"] = start
    grid["end_unix"] = end
    if "day" in market:
        grid["day"] = market["day"]
    cols = [
        "slug", "t", "tau", "y", "winner", "K", "b", "S_open", "start_unix", "end_unix",
        "binance_mid", "coinbase_mid", "S", "sigma", "cb_bn_gap", "binance_signed_vol_10s",
        "up_bid1", "up_bid1_size", "up_ask1", "up_ask1_size", "pm_mid", "imbalance",
        "last_trade", "mid_chg_30",
    ]
    if "day" in grid.columns:
        cols.append("day")
    spot_out = spot[["slug", "sec", "S", "binance_mid", "coinbase_mid", "binance_signed_vol"]].copy()
    return grid[cols].reset_index(drop=True), spot_out


def build_grid(markets, book, trades, btc, half_life_s):
    grids, spots = [], []
    for market in markets.sort_values("start_unix").to_dict(orient="records"):
        slug = market["slug"]
        grid, spot = build_market_grid(
            market,
            btc.loc[btc["slug"] == slug],
            book.loc[book["slug"] == slug],
            trades.loc[trades["slug"] == slug],
            half_life_s,
        )
        grids.append(grid)
        spots.append(spot)
    grid = pd.concat(grids, ignore_index=True)
    spot = pd.concat(spots, ignore_index=True)
    return grid.sort_values(["slug", "t"]).reset_index(drop=True), spot


def attach_sigma(grid, spot, half_life_s):
    """Recompute sigma for a new half-life without rebuilding the rest of the grid."""
    pieces = []
    out = grid.drop(columns=["sigma"]).copy()
    for slug, path in spot.groupby("slug", sort=False):
        path = path.sort_values("sec")
        sigma = ewma_sigma(path["S"].to_numpy(), half_life_s)
        pieces.append(pd.DataFrame({"slug": slug, "t": path["sec"].to_numpy(), "sigma": sigma}))
    sig = pd.concat(pieces, ignore_index=True)
    out["t"] = out["t"].astype("int64")
    sig["t"] = sig["t"].astype("int64")
    out = out.merge(sig, on=["slug", "t"], how="left")
    return out.sort_values(["slug", "t"]).reset_index(drop=True)
