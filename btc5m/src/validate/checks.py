"""Drop markets that fail the integrity checks and write the report.

A market is never dropped for how its price moved. A one-sided book (no ask because
Up is nearly certain, say) is a normal state and is kept.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def failure_reasons(market, limits):
    """Reasons this market cannot be used. An empty list means it is kept.

    `market` is the converter's market row joined with the book and spot statistics.
    """
    reasons = []
    start, end = int(market["start_unix"]), int(market["end_unix"])
    if not bool(market["complete"]):
        reasons.append("incomplete")
    iqr = market["offset_iqr_ms"]
    if not _finite(iqr) or float(iqr) > limits["max_offset_iqr_ms"]:
        reasons.append("clock_iqr")
    if not _finite(market["K"]) or not _finite(market["open_ts"]):
        reasons.append("missing_K")
    elif float(market["open_ts"]) > start + limits["max_strike_delay_s"]:
        reasons.append("late_K")
    elif _finite(market["res_open"]) and abs(float(market["K"]) - float(market["res_open"])) > 0.011:
        reasons.append("strike_mismatch")
    if market["winner"] not in ("up", "down"):
        reasons.append("missing_winner")
    elif _finite(market["res_open"]) and _finite(market["res_close"]):
        official = "up" if float(market["res_close"]) >= float(market["res_open"]) else "down"
        if official != market["winner"]:
            reasons.append("winner_inconsistent")

    if not _finite(market["first_ts"]) or float(market["first_ts"]) > start - limits["min_lead_s"] \
            or float(market["last_ts"]) < end:
        reasons.append("capture_coverage")
    if not _finite(market.get("book_first_ts")) or float(market["book_first_ts"]) > start:
        reasons.append("book_coverage")
    else:
        if float(market["no_book_frac"]) > limits["max_no_book_frac"]:
            reasons.append("no_book")
        if float(market["max_book_gap_s"]) > limits["max_book_gap_s"]:
            reasons.append("book_gap")
    if not _finite(market.get("spot_missing_frac")) or float(market["spot_missing_frac"]) > limits["max_spot_missing_frac"] \
            or not _finite(market["binance_open"]) or not _finite(market["coinbase_open"]):
        reasons.append("spot_coverage")
    return reasons


def validate(markets, pm_stats, spot_table, cfg):
    """Return (kept markets, dropped table, markdown report)."""
    limits = cfg["validation"]
    spot_cols = ["slug", "spot_missing_frac", "binance_open", "binance_close", "coinbase_open", "coinbase_close"]
    table = markets.drop(columns=[c for c in spot_cols[2:] if c in markets.columns])
    table = table.merge(pm_stats, on="slug", how="left").merge(spot_table[spot_cols], on="slug", how="left")
    reasons = [failure_reasons(row, limits) for row in table.to_dict(orient="records")]
    table["reasons"] = [",".join(r) for r in reasons]
    kept = table.loc[table["reasons"] == ""].drop(columns="reasons").reset_index(drop=True)
    dropped = table.loc[table["reasons"] != "", ["slug", "day_idx", "reasons"]].reset_index(drop=True)

    lines = [
        "# Validation report",
        "",
        f"Kept {len(kept)} of {len(table)} markets. Dropped markets are excluded from the grid, "
        "the fit, and the replay.",
        "",
        "A market is dropped when the capture is incomplete or does not span the market, the "
        f"clock-offset IQR exceeds {limits['max_offset_iqr_ms']:g} ms, the strike or the winner is "
        f"missing, the strike was recorded more than {limits['max_strike_delay_s']:g} s after the open, "
        f"the book is entirely empty on more than {limits['max_no_book_frac']:.0%} of seconds or silent "
        f"for more than {limits['max_book_gap_s']:g} s, or the BTC path has holes. "
        "Markets are not dropped for having a one-sided book.",
        "",
    ]
    if len(kept):
        up = (kept["winner"] == "up").mean()
        lines += [f"Up won {up:.1%} of the kept markets.", ""]
        lines += ["Does the exchange price alone pick the official winner (close >= open)?", ""]
        lines += ["| proxy | agrees with the official winner |", "| --- | --- |"]
        for name, o, c in (
            ("Coinbase", kept["coinbase_open"], kept["coinbase_close"]),
            ("Binance", kept["binance_open"], kept["binance_close"]),
            ("mean of the two", (kept["coinbase_open"] + kept["binance_open"]) / 2,
             (kept["coinbase_close"] + kept["binance_close"]) / 2),
        ):
            agree = (np.where(c >= o, "up", "down") == kept["winner"]).mean()
            lines.append(f"| {name} | {agree:.1%} |")
        lines += ["", "Book statistics over the kept markets:", "",
                  f"- median share of seconds with a one-sided book: {kept['one_sided_frac'].median():.1%}",
                  f"- median longest silence in the book: {kept['max_book_gap_s'].median():.1f} s",
                  f"- median taker prints per market: {kept['n_trades_usable'].median():.0f}",
                  f"- strike recorded a median {(kept['open_ts'] - kept['start_unix']).median():.1f} s after the open",
                  ""]
        per_day = kept.groupby("day_idx").size()
        lines += ["Kept markets per day: " + ", ".join(f"{d}: {n}" for d, n in per_day.items()), ""]
    if dropped.empty:
        lines.append("No markets were dropped.")
    else:
        counts = dropped["reasons"].str.split(",").explode().value_counts()
        lines += ["## Dropped", "", "| reason | markets |", "| --- | --- |"]
        lines += [f"| {reason} | {count} |" for reason, count in counts.items()]
        lines += ["", "| slug | day | reasons |", "| --- | --- | --- |"]
        lines += [f"| {r.slug} | {r.day_idx} | {r.reasons} |" for r in dropped.itertuples(index=False)]
    lines.append("")
    return kept, dropped, "\n".join(lines)
