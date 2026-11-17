"""Causal one-second grid built from the converter's Parquet tables.

Each row at second t stores the last observation with timestamp <= t. Latency is
applied later, by reading the row at t - delta, so the grid does not depend on it.

The Polymarket half of the grid is built one market at a time (in parallel) and cached.
The BTC half comes from one continuous per-second path stitched from every capture
file, so volatility has a long history instead of the ~5 minutes in a single file.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import be_nice

LEVELS = ("up_bid1", "up_bid2", "up_ask1", "up_ask2")
BOOK_COLS = [c for level in LEVELS for c in (level, level + "_size")]
PROXIES = ("coinbase", "binance", "mean")
BASIS_WINDOW_S = 1800
PM_COLS = [
    "slug", "t", *BOOK_COLS, "hi_bid_next", "lo_ask_next", "pm_mid", "pm_mid_filled",
    "spread", "imbalance", "mid_chg_30", "last_trade", "pm_flow_30",
]


def load_markets(parquet_dir):
    files = sorted((Path(parquet_dir) / "markets").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {Path(parquet_dir) / 'markets'}")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return frame.sort_values("start_unix").reset_index(drop=True)


# ---------------------------------------------------------------- BTC path

def ewma_sigma(prices, half_life_s):
    """sqrt of an exponentially weighted mean of squared 1-second log returns.

    Only returns at or before each second are used. The weights are normalised by
    their own sum, so the first seconds of a path are an average of what exists
    rather than the first return alone. Before the first return sigma is NaN.
    Units are per sqrt(second).
    """
    s = np.asarray(prices, dtype=float)
    n = len(s)
    squared = np.full(n, np.nan)
    if n >= 2:
        prev, cur = s[:-1], s[1:]
        ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0) & (cur > 0)
        r = np.log(cur[ok] / prev[ok])
        squared[1:][ok] = r * r
    lam = 0.5 ** (1.0 / float(half_life_s))
    var = np.full(n, np.nan)
    num = den = 0.0
    for i, x in enumerate(squared.tolist()):
        if x == x:
            num = lam * num + x
            den = lam * den + 1.0
        if den > 0.0:
            var[i] = num / den
    return np.sqrt(var)


class Spot:
    """Continuous per-second BTC path with cached volatility estimates."""

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.index = pd.Index(self.frame["sec"].to_numpy())
        self._sigma = {}

    def price(self, proxy):
        bn = self.frame["binance_mid"].to_numpy(dtype=float)
        cb = self.frame["coinbase_mid"].to_numpy(dtype=float)
        if proxy == "binance":
            return bn
        if proxy == "coinbase":
            return cb
        if proxy == "mean":
            return (bn + cb) / 2.0
        raise ValueError(f"unknown proxy {proxy!r}")

    def sigma(self, proxy, half_life_s):
        key = (proxy, float(half_life_s))
        if key not in self._sigma:
            price = self.price(proxy)
            out = np.full(len(price), np.nan)
            seg = self.frame["seg"].to_numpy()
            edges = np.flatnonzero(np.diff(seg)) + 1
            for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(price)]):
                out[lo:hi] = ewma_sigma(price[lo:hi], half_life_s)
            self._sigma[key] = out
        return self._sigma[key]

    def at(self, values, secs):
        """values[sec] for each requested second, NaN where the path has no row."""
        pos = self.index.get_indexer(np.asarray(secs, dtype=np.int64))
        out = np.asarray(values, dtype=float)[np.clip(pos, 0, None)]
        return np.where(pos >= 0, out, np.nan)


def build_spot(parquet_dir, markets):
    """Stitch every market's btc_1s table into one path with one row per second.

    Consecutive capture files overlap. A second is taken from the file whose own
    market window contains it, and otherwise from a file where it is not the first
    or last (partial) second.
    """
    root = Path(parquet_dir) / "btc_1s"
    cols = ["sec", "binance_mid", "coinbase_mid", "binance_signed_vol"]
    frames = []
    for slug, start, end in markets[["slug", "start_unix", "end_unix"]].itertuples(index=False):
        path = root / f"{slug}.parquet"
        if not path.exists():
            continue
        f = pd.read_parquet(path, columns=cols)
        if f.empty:
            continue
        sec = f["sec"].to_numpy()
        own = (sec > start) & (sec <= end)
        edge = (sec == sec.min()) | (sec == sec.max())
        f["rank"] = np.where(own, 0, np.where(edge, 2, 1)).astype(np.int8)
        frames.append(f)
    path = pd.concat(frames, ignore_index=True)
    path = path.loc[path["binance_mid"].notna() | path["coinbase_mid"].notna()]
    path = path.sort_values(["sec", "rank"], kind="stable").drop_duplicates("sec", keep="first")
    path = path.drop(columns="rank").reset_index(drop=True)
    path["seg"] = (path["sec"].diff() != 1).cumsum()
    path["binance_signed_vol_10s"] = (
        path.groupby("seg")["binance_signed_vol"].rolling(10, min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )
    return Spot(path)


def market_spot_table(markets, spot):
    """Per market: both venues' mids at the open and close, and the path's coverage of the window."""
    rows = []
    bn = spot.frame["binance_mid"].to_numpy(dtype=float)
    cb = spot.frame["coinbase_mid"].to_numpy(dtype=float)
    for slug, start, end in markets[["slug", "start_unix", "end_unix"]].itertuples(index=False):
        secs = np.arange(start, end + 1, dtype=np.int64)
        b, c = spot.at(bn, secs), spot.at(cb, secs)
        rows.append({
            "slug": slug,
            "binance_open": b[0], "binance_close": b[-1],
            "coinbase_open": c[0], "coinbase_close": c[-1],
            "spot_missing_frac": float(np.mean(~(np.isfinite(b) & np.isfinite(c)))),
        })
    return pd.DataFrame(rows)


def _open_close(table, proxy):
    if proxy == "mean":
        return ((table["binance_open"] + table["coinbase_open"]) / 2.0,
                (table["binance_close"] + table["coinbase_close"]) / 2.0)
    return table[f"{proxy}_open"], table[f"{proxy}_close"]


def basis_table(markets, spot_table, proxy):
    """Settlement price minus the proxy at each market's open and close.

    basis_prior is the median open basis of the markets that started in the half hour
    before this one, all of which are known by the time this market opens.
    """
    t = markets[["slug", "start_unix", "K", "res_close"]].merge(spot_table, on="slug", how="left")
    s_open, s_close = _open_close(t, proxy)
    t["S_open"] = s_open
    t["basis_open"] = t["K"] - s_open
    t["basis_close"] = t["res_close"] - s_close
    t = t.sort_values("start_unix").reset_index(drop=True)
    starts = t["start_unix"].to_numpy()
    basis = t["basis_open"].to_numpy(dtype=float)
    prior = np.full(len(t), np.nan)
    for i, start in enumerate(starts):
        lo = np.searchsorted(starts, start - BASIS_WINDOW_S, side="left")
        hi = np.searchsorted(starts, start, side="left")
        window = basis[lo:hi]
        window = window[np.isfinite(window)]
        if len(window):
            prior[i] = np.median(window)
    t["basis_prior"] = prior
    return t[["slug", "S_open", "basis_open", "basis_close", "basis_prior"]]


def fit_basis(basis, slugs):
    """How much of the open basis is still there at the close, and how noisy the rest is.

    close_basis - prior = beta * (open_basis - prior) + noise, fitted on the given markets.
    beta = 1 means the open basis persists; beta = 0 means it reverts to the recent level.
    The noise scale (in USD) is 1.4826 x the median absolute residual.
    """
    b = basis.loc[basis["slug"].isin(set(slugs))]
    x = (b["basis_open"] - b["basis_prior"]).to_numpy(dtype=float)
    y = (b["basis_close"] - b["basis_prior"]).to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return {"beta": 1.0, "sd": 0.0, "n": int(len(x))}
    scale = 1.4826 * np.median(np.abs(x - np.median(x))) or 1.0
    keep = (np.abs(x) < 8 * scale) & (np.abs(y) < 8 * scale)
    x, y = x[keep], y[keep]
    beta = float(np.clip(np.sum(x * y) / np.sum(x * x), 0.0, 1.0))
    resid = y - beta * x
    sd = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    return {"beta": beta, "sd": sd, "n": int(len(x))}


def attach_spot(pm_grid, markets, spot, basis, proxy, half_life_s, sigma_scale=1.0,
                basis_beta=1.0, basis_sd=0.0):
    """Add the BTC columns the models need to the Polymarket grid.

    The strike is only used from the second it was recorded (open_ts). Before that,
    the row prices the market as if the strike were the proxy at the open.
    """
    out = pm_grid.copy()
    info = markets.merge(basis, on="slug", how="left").set_index("slug")
    slug = out["slug"]
    t = out["t"].to_numpy(dtype=np.int64)
    for col in ("start_unix", "end_unix", "day_idx"):
        out[col] = slug.map(info[col]).to_numpy()
    out["tau"] = out["end_unix"] - out["t"]
    out["y"] = slug.map((info["winner"] == "up").astype(int)).to_numpy()

    price = spot.price(proxy)
    out["binance_mid"] = spot.at(spot.frame["binance_mid"].to_numpy(), t)
    out["coinbase_mid"] = spot.at(spot.frame["coinbase_mid"].to_numpy(), t)
    out["binance_signed_vol_10s"] = spot.at(spot.frame["binance_signed_vol_10s"].to_numpy(), t)
    out["S"] = spot.at(price, t)
    out["sigma"] = spot.at(spot.sigma(proxy, half_life_s), t) * float(sigma_scale)

    s_open = slug.map(info["S_open"]).to_numpy(dtype=float)
    strike = slug.map(info["K"]).to_numpy(dtype=float)
    b_open = slug.map(info["basis_open"]).to_numpy(dtype=float)
    b_prior = slug.map(info["basis_prior"]).to_numpy(dtype=float)
    b_prior = np.where(np.isfinite(b_prior), b_prior, b_open)
    b_eff = basis_beta * b_open + (1.0 - basis_beta) * b_prior
    seen = t >= slug.map(info["open_ts"]).to_numpy(dtype=float)
    out["S_open"] = s_open
    out["K"] = np.where(seen, strike, s_open)
    out["b"] = np.where(seen, b_eff, 0.0)
    out["basis_sd"] = np.where(seen, basis_sd, basis_sd * np.sqrt(2.0))

    gap = out["coinbase_mid"] - out["binance_mid"]
    gap_open = slug.map(info["coinbase_open"] - info["binance_open"]).to_numpy(dtype=float)
    out["gap_chg"] = gap - gap_open
    return out


# ---------------------------------------------------------------- Polymarket side

def _clean_book(book):
    b = book.sort_values("ts", kind="stable").drop_duplicates("ts", keep="last").copy()
    b["ts"] = b["ts"].astype(float)
    for level in LEVELS:
        # Polymarket's tick is 0.001 above 0.96 and below 0.04, so keep three decimals.
        price = b[level].astype(float).round(3)
        missing = ~np.isfinite(price) | (price <= 0)
        b[level] = price.mask(missing)
        b[level + "_size"] = b[level + "_size"].astype(float).mask(missing)
    return b


def usable_trades(trades):
    """Up-mapped taker prints, without the Down leg of a match reported on both tokens."""
    cols = ["ts", "up_side", "up_price", "size"]
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=cols)
    frame = trades
    if "paired" in frame.columns:
        frame = frame.loc[~frame["paired"].fillna(False).astype(bool)]
    frame = frame.loc[frame["up_side"].isin(["buy", "sell"]) & (frame["size"] > 0)]
    return frame[cols].sort_values("ts", kind="stable")


def market_pm_grid(market, book, trades):
    """One market -> (rows for t = start..end, book statistics for the validation report)."""
    slug = market["slug"]
    start, end = int(market["start_unix"]), int(market["end_unix"])
    # Start 30 seconds early so the 30-second mid change is defined at the open.
    secs = np.arange(start - 30, end + 1, dtype=np.int64)
    frame = pd.DataFrame({"t": secs, "t_key": secs.astype(float)})
    b = _clean_book(book)
    m = pd.merge_asof(frame, b[["ts", *BOOK_COLS]], left_on="t_key", right_on="ts",
                      direction="backward")
    bid, ask = m["up_bid1"], m["up_ask1"]
    both = bid.notna() & ask.notna()
    any_side = bid.notna() | ask.notna()
    m["pm_mid"] = np.where(both, (bid + ask) / 2.0, np.nan)
    # A missing side means nobody is quoting there: the bid is 0 or the ask is 1.
    m["pm_mid_filled"] = np.where(any_side, (bid.fillna(0.0) + ask.fillna(1.0)) / 2.0, np.nan)
    m["spread"] = np.where(both, ask - bid, np.nan)
    den = m["up_bid1_size"] + m["up_ask1_size"]
    m["imbalance"] = np.where(both & (den > 0), (m["up_bid1_size"] - m["up_ask1_size"]) / den, np.nan)
    m["mid_chg_30"] = m["pm_mid_filled"] - m["pm_mid_filled"].shift(30)

    # Extremes of the touch during (t, t + 1]. The replay uses them to see whether the
    # other side of the book traded through a resting quote within the second.
    bucket = np.ceil(b["ts"].to_numpy()) - 1
    touch = pd.DataFrame({"t": bucket.astype(np.int64), "bid": b["up_bid1"].to_numpy(),
                          "ask": b["up_ask1"].to_numpy()})
    ext = touch.groupby("t").agg(hi_bid_next=("bid", "max"), lo_ask_next=("ask", "min"))
    m = m.merge(ext, left_on="t", right_index=True, how="left")

    tr = usable_trades(trades)
    if len(tr) == 0:
        m["last_trade"] = np.nan
        m["pm_flow_30"] = 0.0
    else:
        ts = tr["ts"].to_numpy(dtype=float)
        price = tr["up_price"].to_numpy(dtype=float)
        signed = np.where(tr["up_side"].to_numpy() == "buy", 1.0, -1.0) * tr["size"].to_numpy(dtype=float)
        cum = np.r_[0.0, np.cumsum(signed)]
        now = np.searchsorted(ts, secs.astype(float), side="right")
        ago = np.searchsorted(ts, secs.astype(float) - 30.0, side="right")
        m["last_trade"] = np.where(now > 0, price[np.clip(now - 1, 0, None)], np.nan)
        m["pm_flow_30"] = cum[now] - cum[ago]

    grid = m.loc[m["t"] >= start].copy()
    grid["slug"] = slug
    live = grid.loc[grid["t"] < end]
    no_bid, no_ask = live["up_bid1"].isna(), live["up_ask1"].isna()
    stamps = b["ts"].to_numpy()
    inside = stamps[(stamps >= start) & (stamps <= end)]
    gaps = np.diff(np.r_[start, inside, end])
    stats = {
        "slug": slug,
        "book_first_ts": float(stamps.min()) if len(stamps) else np.nan,
        "no_book_frac": float((no_bid & no_ask).mean()),
        "one_sided_frac": float((no_bid ^ no_ask).mean()),
        "max_book_gap_s": float(gaps.max()) if len(gaps) else float(end - start),
        "n_trades_usable": int(((tr["ts"] > start) & (tr["ts"] <= end)).sum()) if len(tr) else 0,
    }
    return grid[PM_COLS].reset_index(drop=True), stats


def _pm_task(args):
    parquet_dir, batch = args
    root = Path(parquet_dir)
    grids, stats = [], []
    for market in batch:
        slug = market["slug"]
        try:
            book = pd.read_parquet(root / "book" / f"{slug}.parquet", columns=["ts", *BOOK_COLS])
            trades = pd.read_parquet(root / "trades" / f"{slug}.parquet")
            g, s = market_pm_grid(market, book, trades)
        except Exception:
            # An unreadable or empty market has no statistics and fails validation.
            stats.append({"slug": slug})
            continue
        grids.append(g)
        stats.append(s)
    grid = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(columns=PM_COLS)
    return grid, stats


def build_pm_grid(parquet_dir, markets, workers=8, cache_dir=None):
    """Polymarket grid and book statistics for every market, cached on disk."""
    slugs = sorted(markets["slug"])
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        grid_path, stats_path = cache_dir / "pm_grid.parquet", cache_dir / "pm_stats.parquet"
        if grid_path.exists() and stats_path.exists():
            stats = pd.read_parquet(stats_path)
            if sorted(stats["slug"]) == slugs:
                return pd.read_parquet(grid_path), stats
    records = markets[["slug", "start_unix", "end_unix"]].to_dict(orient="records")
    batches = [(str(parquet_dir), records[i:i + 32]) for i in range(0, len(records), 32)]
    grids, stats = [], []
    if workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=be_nice) as pool:
            results = list(pool.map(_pm_task, batches))
    else:
        results = [_pm_task(batch) for batch in batches]
    for g, s in results:
        grids.append(g)
        stats.extend(s)
    grid = pd.concat(grids, ignore_index=True).sort_values(["slug", "t"]).reset_index(drop=True)
    stats = pd.DataFrame(stats)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(grid_path, index=False, compression="zstd")
        stats.to_parquet(stats_path, index=False)
    return grid, stats
