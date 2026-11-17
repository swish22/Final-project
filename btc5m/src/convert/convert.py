"""Stream BTC 5-minute capture files into compact per-market Parquet tables.

Each market is downloaded into memory, decoded, written as four small tables and
then dropped, so the raw .bin never touches the disk. Output is about 0.2 MB per market.

usage (from btc5m/):
    python src/convert/convert.py --days 0-13 --per-day 12    # pilot: one hour per day
    python src/convert/convert.py --days 0-13 --workers 8     # the 14 learning days
    python src/convert/convert.py --days 14-16 --workers 8    # the 3 test days
    python src/convert/convert.py --local ../bitcoin4_5_first17d_urls_and_format/raw

Days are 24-hour blocks counted from the first recorded market (day 0 starts
2026-07-10 23:20 UTC). Re-running is safe: converted markets are skipped and
failures are appended to failed.csv in the output folder.

tables (one Parquet file per market in each folder):
    markets/  one row: strike, when the strike was recorded, outcome, clock offset, checks
    trades/   Polymarket trades in exchange time, also expressed as Up-token trades
    book/     top-2 Up book, rows where it changed, from 120 s before the open to 15 s after the end
    btc_1s/   BTC per exchange second for Binance and Coinbase

The Down book is the mirror of the Up book (Up bid = 1 - Down ask), so it is not stored.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import struct
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
TABLES = ("markets", "trades", "book", "btc_1s")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
DAY_S = 86400
BOOK_BEFORE_S = 120
BOOK_AFTER_S = 15

FRAME = struct.Struct("<QBH")
SPOT = struct.Struct("<B3d")
BOOK = struct.Struct("<16d")
TRADE = struct.Struct("<BB3d")
F64 = struct.Struct("<d")
RES = struct.Struct("<B2d")
U64 = struct.Struct("<Q")
UP_COLS = ["up_bid1", "up_bid1_size", "up_bid2", "up_bid2_size",
           "up_ask1", "up_ask1_size", "up_ask2", "up_ask2_size"]
WINNER = {0: "up", 1: "down", 254: "none", 255: "forfeit"}
VENUES = {0: "binance", 1: "coinbase"}


def fetch(url, tries=4):
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except Exception:
            if attempt == tries:
                raise
            time.sleep(5 * attempt)


def decode(raw):
    """Bytes of one capture file -> header + column lists per event type."""
    if raw[:8] not in (b"BT4CAP01", b"BT5CAP01"):
        raise ValueError(f"not a capture file (starts with {raw[:8]!r})")
    hlen = struct.unpack_from("<I", raw, 10)[0]
    header = json.loads(raw[14:14 + hlen])
    d = {"spot": [], "book": [], "trade": [], "btc": [], "open": [], "res": [], "end": None,
         "n": 0, "n_oracle": 0, "truncated": False, "first": math.nan, "last": math.nan}
    off, n = 14 + hlen, len(raw)
    while off < n:
        if off + 11 > n:
            d["truncated"] = True
            break
        wall, ev, plen = FRAME.unpack_from(raw, off)
        p = off + 11
        off = p + plen
        if off > n:
            d["truncated"] = True
            break
        t = wall / 1e9
        if ev == 255:
            d["end"] = U64.unpack_from(raw, p)[0]
            continue
        if d["n"] == 0:
            d["first"] = t
        d["last"] = t
        d["n"] += 1
        if ev == 2:
            d["book"].append((t, *BOOK.unpack_from(raw, p)[:8]))
        elif ev == 1:
            d["spot"].append((t, *SPOT.unpack_from(raw, p)))
        elif ev == 7:
            d["btc"].append((t, *TRADE.unpack_from(raw, p)))
        elif ev == 3:
            d["trade"].append((t, *TRADE.unpack_from(raw, p)))
        elif ev == 4:
            d["open"].append((t, F64.unpack_from(raw, p)[0]))
        elif ev == 6:
            d["res"].append((t, *RES.unpack_from(raw, p)))
        elif ev == 5:
            d["n_oracle"] += 1
    return header, d


def clock_offset(d):
    """Exchange time minus recorder time, from Binance trades (fallback: Coinbase quotes)."""
    diffs = [ts - t for t, v, side, px, sz, ts in d["btc"] if v == 0 and ts > 0]
    if len(diffs) < 4:
        diffs = [ts - t for t, v, ts, b, a in d["spot"] if v == 1 and ts > 0]
    if len(diffs) < 4:
        return math.nan, math.nan
    q1, _, q3 = statistics.quantiles(diffs, n=4)
    return statistics.median(diffs), (q3 - q1) * 1000


def trades_table(slug, d, offset):
    df = pd.DataFrame(d["trade"], columns=["wall", "is_up", "side", "price", "size", "ts"])
    cols = ["ts", "recv_ts", "token", "side", "price", "size", "up_price", "up_side", "paired"]
    if df.empty:
        return pd.DataFrame(columns=cols).assign(slug=slug)
    df["recv_ts"] = df["wall"] + offset
    df["token"] = np.where(df["is_up"] == 1, "up", "down")
    df["side"] = df["side"].map({0: "unknown", 1: "buy", 2: "sell"}).fillna("unknown")
    flip = {"buy": "sell", "sell": "buy", "unknown": "unknown"}
    df["up_price"] = np.where(df["is_up"] == 1, df["price"], 1 - df["price"])
    df["up_side"] = np.where(df["is_up"] == 1, df["side"], df["side"].map(flip))
    # A Down trade with an Up trade of the same size, prices summing to ~1, within 50 ms
    # is one match reported on both tokens. The Down leg is flagged so it is not counted twice.
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["paired"] = False
    up, dn = df[df.is_up == 1], df[df.is_up == 0]
    if len(up) and len(dn):
        m = pd.merge_asof(dn.reset_index().sort_values("ts"), up[["ts", "size", "price"]].sort_values("ts"),
                          on="ts", by="size", tolerance=0.05, direction="nearest", suffixes=("", "_up"))
        hit = m["price_up"].notna() & ((m["price"] + m["price_up"] - 1).abs() < 0.005)
        df.loc[m.loc[hit, "index"], "paired"] = True
    return df[cols].assign(slug=slug)


def book_table(slug, d, offset, start, end):
    df = pd.DataFrame(d["book"], columns=["wall"] + UP_COLS)
    if df.empty:
        return pd.DataFrame(columns=["ts"] + UP_COLS).assign(slug=slug), 0
    df["ts"] = df["wall"] + offset
    n_raw = len(df)
    changed = df[UP_COLS].ne(df[UP_COLS].shift()).any(axis=1)
    df = df[changed]
    # Keep the last state before the window so the book is known at its first second.
    lo, hi = start - BOOK_BEFORE_S, end + BOOK_AFTER_S
    before = df["ts"] < lo
    first_kept = int(before.sum()) - 1
    keep = (df["ts"] >= lo) & (df["ts"] <= hi)
    if first_kept >= 0:
        keep.iloc[first_kept] = True
    df = df[keep].copy()
    df[UP_COLS] = df[UP_COLS].astype("float32")
    return df[["ts"] + UP_COLS].assign(slug=slug), n_raw


def btc_1s_table(slug, d, offset, end):
    """One row per exchange second `sec`; values are the last ones seen at or before `sec`."""
    q = pd.DataFrame(d["spot"], columns=["wall", "v", "src", "bid", "ask"])
    q["t"] = np.where((q["v"] == 1) & (q["src"] > 0), q["src"], q["wall"] + offset)  # Binance quotes lack src time
    q["mid"] = (q["bid"] + q["ask"]) / 2
    tr = pd.DataFrame(d["btc"], columns=["wall", "v", "aggr", "price", "size", "src"])
    tr["t"] = np.where(tr["src"] > 0, tr["src"], tr["wall"] + offset)
    tr["signed"] = np.where(tr["aggr"] == 1, tr["size"], np.where(tr["aggr"] == 2, -tr["size"], 0.0))
    times = pd.concat([q["t"], tr["t"]])
    if times.empty:
        return pd.DataFrame({"sec": pd.Series(dtype="int64")}).assign(slug=slug)
    secs = np.arange(math.ceil(times.min()), min(math.ceil(times.max()), end + 60) + 1)
    out = pd.DataFrame({"sec": secs})
    for code, name in VENUES.items():
        qq = q[q["v"] == code].assign(sec=lambda x: np.ceil(x["t"]).astype("int64"))
        mid = qq.sort_values("t", kind="stable").groupby("sec")["mid"].last()
        out[f"{name}_mid"] = mid.reindex(secs).ffill().to_numpy()
        tt = tr[tr["v"] == code].assign(sec=lambda x: np.ceil(x["t"]).astype("int64"))
        g = tt.sort_values("t", kind="stable").groupby("sec").agg(
            n=("size", "size"), vol=("size", "sum"), signed=("signed", "sum"), last=("price", "last"))
        g = g.reindex(secs)
        out[f"{name}_trades"] = g["n"].fillna(0).astype("int32").to_numpy()
        out[f"{name}_vol"] = g["vol"].fillna(0).to_numpy()
        out[f"{name}_signed_vol"] = g["signed"].fillna(0).to_numpy()
        out[f"{name}_last"] = g["last"].ffill().to_numpy()
    return out.assign(slug=slug)


def _at(series, sec):
    value = series.get(sec, math.nan)
    return float(value) if value is not None else math.nan


def convert_one(source, out_dir, first_start):
    """source = URL or local path. Writes the four tables; returns a short status line."""
    t0 = time.time()
    raw = open(source, "rb").read() if os.path.exists(source) else fetch(source)
    header, d = decode(raw)
    raw_mb = len(raw) / 1e6
    del raw
    slug = header["slug"]
    start, end = int(header["start_unix"]), int(header["end_unix"])
    offset, iqr_ms = clock_offset(d)
    if math.isnan(offset):
        raise ValueError("no exchange timestamps to align the clock")

    trades = trades_table(slug, d, offset)
    book, n_book_raw = book_table(slug, d, offset, start, end)
    btc = btc_1s_table(slug, d, offset, end)

    open_wall, K = d["open"][0] if d["open"] else (math.nan, math.nan)
    res_wall, w, r_open, r_close = d["res"][0] if d["res"] else (math.nan, None, math.nan, math.nan)
    by_sec = btc.set_index("sec") if len(btc) else pd.DataFrame()
    mids = {}
    for venue in ("binance", "coinbase"):
        col = by_sec[f"{venue}_mid"] if f"{venue}_mid" in by_sec else pd.Series(dtype=float)
        mids[venue] = (_at(col, start), _at(col, end))

    def proxy(venue):
        m0, m1 = mids[venue]
        return "unknown" if math.isnan(m0) or math.isnan(m1) else ("up" if m1 >= m0 else "down")

    winner = WINNER.get(w, "unknown" if w is None else str(w))
    in_window = trades[(trades["ts"] >= start) & (trades["ts"] <= end)] if len(trades) else trades
    market = pd.DataFrame([{
        "slug": slug, "start_unix": start, "end_unix": end,
        "day_idx": (start - first_start) // DAY_S,
        "K": K, "open_ts": open_wall + offset,
        "winner": winner, "res_open": r_open, "res_close": r_close, "res_ts": res_wall + offset,
        "complete": (not d["truncated"]) and d["end"] == d["n"],
        "n_frames": d["n"], "n_oracle": d["n_oracle"], "raw_mb": round(raw_mb, 1),
        "first_ts": d["first"] + offset, "last_ts": d["last"] + offset,
        "offset_s": offset, "offset_iqr_ms": iqr_ms,
        "n_trades": len(trades), "n_trades_window": len(in_window),
        "n_paired": int(trades["paired"].sum()) if len(trades) else 0,
        "n_book_raw": n_book_raw, "n_book_kept": len(book),
        "binance_open": mids["binance"][0], "binance_close": mids["binance"][1],
        "coinbase_open": mids["coinbase"][0], "coinbase_close": mids["coinbase"][1],
        "proxy_winner": proxy("binance"), "proxy_winner_cb": proxy("coinbase"),
    }])
    del d

    # The markets row is written last: its presence is what marks a market as converted.
    for name, df in (("trades", trades), ("book", book), ("btc_1s", btc), ("markets", market)):
        tmp = os.path.join(out_dir, name, slug + ".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, os.path.join(out_dir, name, slug + ".parquet"))
    out_mb = sum(os.path.getsize(os.path.join(out_dir, t, slug + ".parquet")) for t in TABLES) / 1e6
    return slug, f"{raw_mb:.0f} MB -> {out_mb:.2f} MB in {time.time() - t0:.0f} s"


def job(label, source, out_dir, first_start):
    try:
        slug, msg = convert_one(source, out_dir, first_start)
        return slug, True, msg
    except Exception as e:
        return label, False, f"{type(e).__name__}: {e}"


def _be_nice():
    """Workers run below normal priority so a long download leaves the machine usable."""
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


def pick_rows(rows, first_start, days, per_day):
    lo, hi = days
    chosen = []
    for day in range(lo, hi + 1):
        of_day = [r for r in rows if (int(r["start_unix"]) - first_start) // DAY_S == day]
        if per_day and len(of_day) > per_day:
            # A block of consecutive markets, starting at a different hour each day.
            begin = (day * 7 % 24) * 12
            begin = min(begin, len(of_day) - per_day)
            of_day = of_day[begin:begin + per_day]
        chosen.extend(of_day)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--days", help="day-index range, e.g. 0-13")
    g.add_argument("--local", metavar="DIR", help="convert .bin files already in DIR")
    ap.add_argument("--per-day", type=int, help="only a block of N consecutive markets per day")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if os.name == "nt":
        # Ask Windows not to sleep while this runs. The request ends with the process.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    index = (ROOT / cfg["paths"]["index"]).resolve()
    out_dir = str((ROOT / cfg["paths"]["parquet_dir"]).resolve())
    for t in TABLES:
        os.makedirs(os.path.join(out_dir, t), exist_ok=True)
    rows = list(csv.DictReader(open(index)))
    first_start = int(rows[0]["start_unix"])

    done = {os.path.basename(p)[:-8] for p in glob.glob(os.path.join(out_dir, "markets", "*.parquet"))}
    if args.local:
        jobs = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(os.path.join(args.local, "*.bin")))]
    else:
        lo, _, hi = args.days.partition("-")
        picked = pick_rows(rows, first_start, (int(lo), int(hi or lo)), args.per_day)
        jobs = [(r["slug"], r["url"]) for r in picked]
    total = len(jobs)
    jobs = [j for j in jobs if j[0] not in done]
    if not jobs:
        sys.exit(f"nothing to do ({total} selected, all already converted)")
    print(f"{len(jobs)} markets to convert with {args.workers} workers "
          f"({total - len(jobs)} already done, skipped)", flush=True)

    failed, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_be_nice) as pool:
        futures = [pool.submit(job, s, src, out_dir, first_start) for s, src in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            slug, ok, msg = f.result()
            if not ok or i % 25 == 0 or i == len(jobs):
                mins = (time.time() - t0) / 60
                print(f"[{i}/{len(jobs)}] {slug} {'ok' if ok else 'FAILED'}: {msg}  ({mins:.1f} min)", flush=True)
            if not ok:
                failed.append((slug, msg))
    if failed:
        path = os.path.join(out_dir, "failed.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["slug", "error"])
            w.writerows(failed)
    size = sum(os.path.getsize(p) for t in TABLES for p in glob.glob(os.path.join(out_dir, t, "*.parquet")))
    print(f"done in {(time.time() - t0) / 60:.1f} min: {len(jobs) - len(failed)} ok, {len(failed)} failed"
          + (" (re-run the same command to retry)" if failed else "")
          + f"; output folder holds {size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
