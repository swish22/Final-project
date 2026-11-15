"""Stream BTC 5-minute capture files into small Parquet tables, never saving the raw .bin.

Each market is downloaded into memory, decoded, written as small tables under out/,
and then dropped. Disk use is only the output, roughly a few MB per market.

usage:
    python3 convert.py --day 2026-07-11 --limit 5      # test: first 5 markets of Jul 11
    python3 convert.py --day 2026-07-11 --workers 4    # one full day, 4 at a time
    python3 convert.py --all --workers 4               # all 4,895 markets (run overnight)
    python3 convert.py --local raw                      # convert .bin files you already downloaded

Re-running is safe: markets already in out/markets/ are skipped, failures go to failed.csv.

needs:  pip install pandas pyarrow
output (one Parquet file per market in each folder; read a whole folder with pd.read_parquet("out/trades")):
    out/markets/  one row per market: strike, outcome, clock offset, data checks
    out/trades/   every Polymarket trade, exchange time, also expressed as an Up-token trade
    out/book/     top-2 Polymarket book, consecutive duplicates removed, exchange time
    out/btc_1s/   BTC per second for Binance and Coinbase: mid, trade count, volume, signed volume
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

import numpy as np
import pandas as pd

INDEX = "bitcoin4_first17d_index.csv"
OUT = "out"
TABLES = ("markets", "trades", "book", "btc_1s")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

FRAME = struct.Struct("<QBH")
SPOT = struct.Struct("<B3d")
BOOK = struct.Struct("<16d")
TRADE = struct.Struct("<BB3d")
F64 = struct.Struct("<d")
RES = struct.Struct("<B2d")
U64 = struct.Struct("<Q")
BOOK_COLS = ["up_bid1", "up_bid1_size", "up_bid2", "up_bid2_size", "up_ask1", "up_ask1_size", "up_ask2",
             "up_ask2_size", "down_bid1", "down_bid1_size", "down_bid2", "down_bid2_size", "down_ask1",
             "down_ask1_size", "down_ask2", "down_ask2_size"]
WINNER = {0: "up", 1: "down", 254: "none", 255: "forfeit"}
VENUES = {0: "binance", 1: "coinbase"}


# ---------------------------------------------------------------- download and decode

def fetch(url, tries=3):
    """Download url into memory (bytes). Raises after the last failed try."""
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception:
            if attempt == tries:
                raise
            time.sleep(3 * attempt)


def decode(raw):
    """Bytes of one capture file -> header + column lists per event type (no per-frame dicts)."""
    if raw[:8] not in (b"BT4CAP01", b"BT5CAP01"):
        raise ValueError(f"not a capture file (starts with {raw[:8]!r})")
    hlen = struct.unpack_from("<I", raw, 10)[0]
    header = json.loads(raw[14:14 + hlen])
    d = {"spot": [], "book": [], "trade": [], "btc": [], "open": [], "res": [], "end": None, "n": 0,
         "truncated": False}
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
        d["n"] += 1
        if ev == 1:
            d["spot"].append((t, *SPOT.unpack_from(raw, p)))
        elif ev == 2:
            d["book"].append((t, *BOOK.unpack_from(raw, p)))
        elif ev == 3:
            d["trade"].append((t, *TRADE.unpack_from(raw, p)))
        elif ev == 7:
            d["btc"].append((t, *TRADE.unpack_from(raw, p)))
        elif ev == 4:
            d["open"].append(F64.unpack_from(raw, p)[0])
        elif ev == 6:
            d["res"].append(RES.unpack_from(raw, p))
    return header, d


# ---------------------------------------------------------------- build tables

def clock_offset(d):
    """Exchange time minus recorder time, from Binance trades (fallback: Coinbase quotes)."""
    diffs = [ts - t for t, v, side, px, sz, ts in d["btc"] if v == 0 and ts > 0]
    if len(diffs) < 4:
        diffs = [ts - t for t, v, ts, b, a in d["spot"] if v == 1 and ts > 0]
    if len(diffs) < 4:
        return math.nan, math.nan
    q1, _, q3 = statistics.quantiles(diffs, n=4)
    return statistics.median(diffs), (q3 - q1) * 1000


def trades_table(slug, d):
    df = pd.DataFrame(d["trade"], columns=["wall", "is_up", "side", "price", "size", "ts"])
    if df.empty:
        return df.assign(slug=slug)
    df["token"] = np.where(df["is_up"] == 1, "up", "down")
    df["side"] = df["side"].map({0: "unknown", 1: "buy", 2: "sell"})
    flip = {"buy": "sell", "sell": "buy", "unknown": "unknown"}
    df["up_price"] = np.where(df["is_up"] == 1, df["price"], 1 - df["price"])
    df["up_side"] = np.where(df["is_up"] == 1, df["side"], df["side"].map(flip))
    # a Down trade with an Up trade of the same size, prices summing to ~1, within 50 ms = one paired match
    df = df.sort_values("ts").reset_index(drop=True)
    df["paired"] = False
    up, dn = df[df.is_up == 1], df[df.is_up == 0]
    if len(up) and len(dn):
        m = pd.merge_asof(dn.reset_index().sort_values("ts"), up[["ts", "size", "price"]].sort_values("ts"),
                          on="ts", by="size", tolerance=0.05, direction="nearest", suffixes=("", "_up"))
        hit = m["price_up"].notna() & ((m["price"] + m["price_up"] - 1).abs() < 0.005)
        df.loc[m.loc[hit, "index"], "paired"] = True
    return df[["ts", "token", "side", "price", "size", "up_price", "up_side", "paired"]].assign(slug=slug)


def book_table(slug, d, offset):
    df = pd.DataFrame(d["book"], columns=["wall"] + BOOK_COLS)
    if df.empty:
        return df.assign(ts=[], slug=slug)
    changed = df[BOOK_COLS].ne(df[BOOK_COLS].shift()).any(axis=1)    # drop consecutive duplicates
    df = df[changed].copy()
    df["ts"] = df["wall"] + offset
    df[BOOK_COLS] = df[BOOK_COLS].astype("float32")
    return df[["ts"] + BOOK_COLS].assign(slug=slug)


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
        return pd.DataFrame({"slug": []})
    secs = np.arange(math.ceil(times.min()), min(math.ceil(times.max()), end + 60) + 1)
    out = pd.DataFrame({"sec": secs})
    for code, name in VENUES.items():
        qq = q[q["v"] == code].assign(sec=lambda x: np.ceil(x["t"]).astype("int64"))
        mid = qq.groupby("sec")["mid"].last()
        out[f"{name}_mid"] = mid.reindex(secs).ffill().to_numpy()
        tt = tr[tr["v"] == code].assign(sec=lambda x: np.ceil(x["t"]).astype("int64"))
        g = tt.groupby("sec").agg(n=("size", "size"), vol=("size", "sum"), signed=("signed", "sum"),
                                  last=("price", "last"))
        g = g.reindex(secs)
        out[f"{name}_trades"] = g["n"].fillna(0).astype("int32").to_numpy()
        out[f"{name}_vol"] = g["vol"].fillna(0).to_numpy()
        out[f"{name}_signed_vol"] = g["signed"].fillna(0).to_numpy()
        out[f"{name}_last"] = g["last"].ffill().to_numpy()
    return out.assign(slug=slug)


def convert_one(slug, source, raw_bytes=None):
    """source = URL or local path. Writes the four tables; returns a short status line."""
    t0 = time.time()
    raw = raw_bytes if raw_bytes is not None else (
        open(source, "rb").read() if os.path.exists(source) else fetch(source))
    header, d = decode(raw)
    slug = header["slug"]
    start, end = header["start_unix"], header["end_unix"]
    offset, iqr_ms = clock_offset(d)
    if math.isnan(offset):
        raise ValueError("no exchange timestamps to align the clock")

    trades = trades_table(slug, d)
    book = book_table(slug, d, offset)
    btc = btc_1s_table(slug, d, offset, end)

    K = d["open"][0] if d["open"] else math.nan
    w, r_open, r_close = d["res"][0] if d["res"] else (None, math.nan, math.nan)
    bm = btc.set_index("sec")["binance_mid"] if "binance_mid" in btc else pd.Series(dtype=float)
    m0, m1 = bm.get(start, math.nan), bm.get(end, math.nan)
    proxy = None if math.isnan(m0) or math.isnan(m1) else ("up" if m1 >= m0 else "down")
    market = pd.DataFrame([{
        "slug": slug, "start_unix": start, "end_unix": end, "K": K,
        "winner": WINNER.get(w, w), "res_open": r_open, "res_close": r_close,
        "complete": (not d["truncated"]) and d["end"] == d["n"],
        "n_frames": d["n"], "raw_mb": round(len(raw) / 1e6, 1),
        "offset_s": offset, "offset_iqr_ms": iqr_ms,
        "n_trades": len(trades), "n_paired": int(trades["paired"].sum()) if len(trades) else 0,
        "n_book_raw": len(d["book"]), "n_book_kept": len(book),
        "basis_binance": K - m0, "proxy_winner": proxy, "proxy_agrees": proxy == WINNER.get(w, w),
    }])
    del raw, d                                   # free the big objects before writing

    for name, df in zip(TABLES, (market, trades, book, btc)):
        tmp = os.path.join(OUT, name, slug + ".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, os.path.join(OUT, name, slug + ".parquet"))
    return f"{market.raw_mb[0]:.0f} MB -> {out_mb(slug):.1f} MB in {time.time() - t0:.0f} s"


def out_mb(slug):
    return sum(os.path.getsize(os.path.join(OUT, t, slug + ".parquet")) for t in TABLES) / 1e6


def job(slug, source):
    try:
        return slug, True, convert_one(slug, source)
    except Exception as e:
        return slug, False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--day", help="UTC day, e.g. 2026-07-11")
    g.add_argument("--all", action="store_true", help="every market in the index")
    g.add_argument("--local", metavar="DIR", help="convert .bin files already in DIR (keeps them)")
    ap.add_argument("--limit", type=int, help="only the first N markets")
    ap.add_argument("--workers", type=int, default=2, help="markets processed at once (default 2)")
    args = ap.parse_args()

    for t in TABLES:
        os.makedirs(os.path.join(OUT, t), exist_ok=True)

    if args.local:
        jobs = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(os.path.join(args.local, "*.bin")))]
        done = set()                            # local files: slug only known after decoding
    else:
        rows = list(csv.DictReader(open(INDEX)))
        if args.day:
            rows = [r for r in rows if r["start_utc"].startswith(args.day)]
        jobs = [(r["slug"], r["url"]) for r in rows]
        done = {os.path.basename(p)[:-8] for p in glob.glob(os.path.join(OUT, "markets", "*.parquet"))}
    jobs = [j for j in jobs if j[0] not in done][:args.limit]
    if not jobs:
        sys.exit("nothing to do (no matching markets, or all already converted)")
    print(f"{len(jobs)} markets to convert with {args.workers} workers "
          f"({len(done)} already done, skipped)")

    failed, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, s, src) for s, src in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            slug, ok, msg = f.result()
            print(f"[{i}/{len(jobs)}] {slug} {'ok' if ok else 'FAILED'}: {msg}", flush=True)
            if not ok:
                failed.append((slug, msg))
    if failed:
        new = not os.path.exists("failed.csv")
        with open("failed.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["slug", "error"])
            w.writerows(failed)
    mins = (time.time() - t0) / 60
    print(f"done in {mins:.1f} min: {len(jobs) - len(failed)} ok, {len(failed)} failed"
          + (" (see failed.csv; re-run the same command to retry)" if failed else ""))
    total = sum(os.path.getsize(p) for t in TABLES for p in glob.glob(os.path.join(OUT, t, "*.parquet")))
    print(f"out/ now holds {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()