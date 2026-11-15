"""Drop markets that fail the README integrity checks and write the report."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import utc_day


def _missing_touch_fraction(book, start, end):
    """Share of market seconds whose last book has no Up bid or no Up ask."""
    if book.empty:
        return 1.0
    b = book.sort_values("ts").copy()
    for col in ("up_bid1", "up_ask1"):
        b.loc[~np.isfinite(b[col].astype(float)) | (b[col].astype(float) < 0), col] = np.nan
    secs = pd.DataFrame({"t": np.arange(int(start), int(end) + 1, dtype=np.int64)})
    secs["t"] = secs["t"].astype(float)
    b["ts"] = b["ts"].astype(float)
    merged = pd.merge_asof(secs, b[["ts", "up_bid1", "up_ask1"]], left_on="t", right_on="ts",
                           direction="backward")
    missing = merged["up_bid1"].isna() | merged["up_ask1"].isna()
    return float(missing.mean())


def _finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def failure_reasons(market, book, btc, max_iqr_ms, max_missing_book_frac):
    """Reasons this market cannot be used. An empty list means it is kept."""
    reasons = []
    if not bool(market["complete"]):
        reasons.append("incomplete")
    iqr = market["offset_iqr_ms"]
    if not _finite_number(iqr) or float(iqr) > max_iqr_ms:
        reasons.append("clock_iqr")
    k = market["K"]
    if not _finite_number(k):
        reasons.append("missing_K")
    if market["winner"] not in ("up", "down"):
        reasons.append("missing_winner")

    start, end = int(market["start_unix"]), int(market["end_unix"])
    if book.empty or float(book["ts"].min()) > start or float(book["ts"].max()) < end:
        reasons.append("book_coverage")
    if btc.empty or int(btc["sec"].min()) > start or int(btc["sec"].max()) < end:
        reasons.append("btc_coverage")
    else:
        # A proxy mid has to exist at the open so the digital basis is defined.
        at_open = btc.loc[btc["sec"] <= start]
        if at_open.empty or (
            not np.isfinite(at_open["binance_mid"].iloc[-1])
            and not np.isfinite(at_open["coinbase_mid"].iloc[-1])
        ):
            reasons.append("no_proxy_at_open")

    if "book_coverage" not in reasons:
        missing = _missing_touch_fraction(book, start, end)
        if missing > max_missing_book_frac:
            reasons.append(f"missing_touch:{missing:.0%}")
    return reasons


def validate(markets, book, btc, cfg):
    """Return (kept markets, dropped table, markdown report)."""
    max_iqr = float(cfg["validation"]["max_offset_iqr_ms"])
    max_missing = float(cfg["validation"]["max_missing_book_frac"])
    kept_rows = []
    dropped_rows = []
    for market in markets.sort_values("start_unix").to_dict(orient="records"):
        slug = market["slug"]
        book_m = book.loc[book["slug"] == slug]
        btc_m = btc.loc[btc["slug"] == slug]
        reasons = failure_reasons(market, book_m, btc_m, max_iqr, max_missing)
        day = utc_day(market["start_unix"])
        if reasons:
            dropped_rows.append({"slug": slug, "day": day, "reasons": ",".join(reasons)})
        else:
            row = dict(market)
            row["day"] = day
            kept_rows.append(row)

    kept = pd.DataFrame(kept_rows)
    dropped = pd.DataFrame(dropped_rows)
    if not kept.empty and "proxy_agrees" in kept.columns:
        agrees = int(kept["proxy_agrees"].sum())
        rate = agrees / len(kept)
    else:
        agrees, rate = 0, float("nan")

    lines = [
        "# Validation report",
        "",
        f"Kept {len(kept)} of {len(markets)} markets. "
        f"Dropped markets are excluded from the grid, the fit, and the replay.",
        "",
        f"Binance proxy winner matches the official winner in {agrees}/{len(kept)} kept markets"
        + (f" ({rate:.0%})." if kept.shape[0] else "."),
        "",
        "A market is dropped when the capture is incomplete, the clock-offset IQR exceeds "
        f"{max_iqr:g} ms, the strike or the winner is missing, the book or the 1-second BTC "
        "path does not cover `[start, end]`, there is no spot mid at the open, or the Up "
        f"bid/ask is missing on more than {max_missing:.0%} of market seconds.",
        "",
    ]
    if dropped.empty:
        lines.append("No markets were dropped.")
    else:
        lines.append("## Dropped")
        lines.append("")
        lines.append("| slug | day | reasons |")
        lines.append("| --- | --- | --- |")
        for row in dropped.itertuples(index=False):
            lines.append(f"| {row.slug} | {row.day} | {row.reasons} |")
    lines.append("")
    if not kept.empty:
        lines.append("## Kept")
        lines.append("")
        lines.append("| slug | day | winner | proxy agrees | offset IQR ms |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in kept.sort_values("start_unix").itertuples(index=False):
            lines.append(
                f"| {row.slug} | {row.day} | {row.winner} | {bool(row.proxy_agrees)} | "
                f"{row.offset_iqr_ms:.2f} |"
            )
        lines.append("")
    return kept, dropped, "\n".join(lines) + "\n"
