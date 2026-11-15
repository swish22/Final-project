"""One summary row per downloaded market, as a first data check.

usage:
    python3 summarize.py            # every .bin in raw/
    python3 summarize.py raw/btc-updown-5m-1783728000.bin

Prints a table and writes summary.csv. Per market it reports:
  complete       file ends with the END frame and its frame count matches
  trades, books  number of Polymarket trades and order-book updates
  K              price to beat (OPEN event)
  winner, res_open, res_close   official result (RESOLUTION event)
  offset_s       recorder clock correction: median(Binance trade time - recorder time)
  offset_iqr_ms  how much that correction wobbles within the file (want < 50 ms)
  basis_usd      K minus the Binance mid at the market's start
  proxy_winner   winner implied by Binance alone (mid at end vs mid at start)
  agree          does proxy_winner match the official winner?
"""
import glob
import statistics
import sys

import pandas as pd

from read_capture import read_capture

files = sys.argv[1:] or sorted(glob.glob("raw/*.bin"))
if not files:
    sys.exit("no .bin files found; run download.py first")


def quartiles(xs):
    q = statistics.quantiles(xs, n=4)
    return q[0], q[2]


def mid_at(spots, t):
    """Binance mid from the last quote at or before time t (exchange time)."""
    before = [m for ts, m in spots if ts <= t]
    return before[-1] if before else float("nan")


rows = []
for path in files:
    _, _, header, frames = read_capture(path)
    start, end = header["start_unix"], header["end_unix"]
    last = frames[-1]
    complete = last[1] == 255 and last[2]["frame_count"] == len(frames) - 1

    # clock offset from Binance trades, which carry exchange time
    diffs = [f["source_ts"] - t / 1e9 for t, e, f in frames
             if e == 7 and f["venue"] == "binance" and f["source_ts"] > 0]
    offset = statistics.median(diffs) if diffs else float("nan")
    iqr_ms = (lambda q: (q[1] - q[0]) * 1000)(quartiles(diffs)) if len(diffs) >= 4 else float("nan")

    # Binance quotes have no exchange time, so shift recorder time by the offset
    spots = [(t / 1e9 + offset, (f["bid"] + f["ask"]) / 2) for t, e, f in frames
             if e == 1 and f["venue"] == "binance"]
    m0, m1 = mid_at(spots, start), mid_at(spots, end)

    K = next((f["price_to_beat"] for t, e, f in frames if e == 4), float("nan"))
    res = next((f for t, e, f in frames if e == 6), {})
    proxy = "up" if m1 >= m0 else "down"

    rows.append({
        "slug": header["slug"],
        "complete": complete,
        "trades": sum(1 for _, e, _ in frames if e == 3),
        "books": sum(1 for _, e, _ in frames if e == 2),
        "K": K,
        "winner": res.get("winner"),
        "res_open": res.get("open"),
        "res_close": res.get("close"),
        "offset_s": round(offset, 3),
        "offset_iqr_ms": round(iqr_ms, 1),
        "basis_usd": round(K - m0, 2),
        "proxy_winner": proxy,
        "agree": proxy == res.get("winner"),
    })
    print("read", header["slug"], "complete" if complete else "INCOMPLETE")

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print()
print(df.to_string(index=False))
print(f"\n{df['complete'].sum()}/{len(df)} complete; Binance picks the official winner in "
      f"{df['agree'].mean():.0%} of markets")
df.to_csv("summary.csv", index=False)
print("wrote summary.csv")