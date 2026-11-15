"""Replay a quoter through one market's recorded book and Up-mapped trades.

The quote posted at second t is built from the grid row at t - delta. Trades in
(t, t + 1] can fill it. A taker sell hits the bid; a taker buy hits the ask.
Paired Down-leg prints are skipped because the matching Up print is already in
the trade list.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import cents
from mm.queue_fill import apply_cancel, apply_trade, level_size, queue_ahead

_DECISION = {
    "p": "p_dec",
    "dp_dS": "dp_dec",
    "sigma": "sigma_dec",
    "S": "S_dec",
    "up_bid1": "bid_dec",
    "up_ask1": "ask_dec",
    "up_bid1_size": "bid_sz_dec",
    "up_ask1_size": "ask_sz_dec",
}

_BOOK_FIELDS = (
    "up_bid1", "up_bid1_size", "up_bid2", "up_bid2_size",
    "up_ask1", "up_ask1_size", "up_ask2", "up_ask2_size",
)


def replay(grid, book, trades, quoter, fill_rule, delta_s, book_by=None, trades_by=None):
    """Replay every market. Returns (fills, path) dataframes."""
    if book_by is None:
        book_by = _by_slug(book)
    if trades_by is None:
        trades_by = _by_slug(trades)
    fills, paths = [], []
    empty_book = pd.DataFrame(columns=["slug", "ts", *_BOOK_FIELDS])
    empty_trades = pd.DataFrame(columns=["slug", "ts", "up_side", "up_price", "size", "paired"])
    for slug, frame in grid.groupby("slug", sort=False):
        market_fills, market_path = replay_market(
            frame,
            book_by.get(slug, empty_book),
            trades_by.get(slug, empty_trades),
            quoter,
            fill_rule,
            delta_s,
        )
        fills.append(market_fills)
        paths.append(market_path)
    return _concat(fills), _concat(paths)


def replay_market(grid_m, book_m, trades_m, quoter, fill_rule, delta_s):
    frame = grid_m.sort_values("t").reset_index(drop=True)
    lagged = frame[list(_DECISION)].shift(delta_s) if delta_s else frame[list(_DECISION)]
    for src, dst in _DECISION.items():
        frame[dst] = lagged[src].to_numpy()

    trade_groups = _groups(_usable_trades(trades_m), "ts")
    book_groups = _groups(book_m, "ts") if fill_rule == "generous" and book_m is not None and len(book_m) else {}

    bid = _blank_order()
    ask = _blank_order()
    cash = 0.0
    inventory = 0.0
    end = int(frame["end_unix"].iloc[0])
    payoff = 1.0 if int(frame["y"].iloc[0]) == 1 else 0.0
    slug = frame["slug"].iloc[0]
    split = frame["split"].iloc[0] if "split" in frame.columns else None
    last_mid = np.nan
    fills, path = [], []

    for row in frame.itertuples(index=False):
        t = int(row.t)
        if t >= end:
            break
        book_view = {
            "up_bid1": row.bid_dec,
            "up_bid1_size": row.bid_sz_dec,
            "up_ask1": row.ask_dec,
            "up_ask1_size": row.ask_sz_dec,
            "S": row.S_dec,
            "dp_dS": row.dp_dec,
        }
        bid_px, ask_px, bid_sz, ask_sz = quoter.quote(
            row.p_dec, row.sigma_dec, row.tau, inventory, book_view,
        )
        _rest(bid, bid_px, bid_sz, book_view, "bid")
        _rest(ask, ask_px, ask_sz, book_view, "ask")

        events = [(float(rec.ts), 0, "trade", rec) for rec in trade_groups.get(t, ())]
        if fill_rule == "generous":
            events.extend((float(rec.ts), 1, "book", rec) for rec in book_groups.get(t, ()))
        events.sort(key=lambda item: (item[0], item[1]))

        for ts, _, kind, rec in events:
            if kind == "book":
                _on_book(bid, rec, "bid")
                _on_book(ask, rec, "ask")
                mid = _mid(rec)
                if mid is not None:
                    last_mid = mid
                continue
            side_name = "bid" if rec.up_side == "sell" else "ask"
            state = bid if side_name == "bid" else ask
            if state["price"] is not None and cents(rec.up_price) == cents(state["price"]):
                state["traded"] += float(rec.size)
            queue, size, fill = apply_trade(
                state["queue"], state["size"], rec.up_price, rec.size, state["price"], side_name,
            )
            state["queue"] = queue
            state["size"] = size
            if fill <= 0 or state["price"] is None:
                continue
            if side_name == "bid":
                cash -= state["price"] * fill
                inventory += fill
                label = "buy"
            else:
                cash += state["price"] * fill
                inventory -= fill
                label = "sell"
            fills.append({
                "slug": slug,
                "split": split,
                "t": t,
                "ts": ts,
                "side": label,
                "price": state["price"],
                "size": fill,
                "y": payoff,
                "inventory": inventory,
                "cash": cash,
            })

        if _finite(row.pm_mid):
            last_mid = float(row.pm_mid)
        mark = last_mid if _finite(last_mid) else (float(row.p_dec) if _finite(row.p_dec) else 0.0)
        path.append({
            "slug": slug,
            "split": split,
            "t": t,
            "bid": bid["price"],
            "ask": ask["price"],
            "inventory": inventory,
            "cash": cash,
            "mid": mark,
            "p": row.p_dec if _finite(row.p_dec) else np.nan,
            "equity": cash + inventory * mark,
            "settled": False,
        })

    path.append({
        "slug": slug,
        "split": split,
        "t": end,
        "bid": None,
        "ask": None,
        "inventory": inventory,
        "cash": cash,
        "mid": payoff,
        "p": np.nan,
        "equity": cash + inventory * payoff,
        "settled": True,
    })
    return pd.DataFrame(fills), pd.DataFrame(path)


def _rest(state, price, size, book_view, side):
    if price is None or size <= 1e-12:
        state.update(price=None, size=0.0, queue=0.0, prev=None, traded=0.0)
        return
    same = (
        state["price"] is not None
        and cents(state["price"]) == cents(price)
        and state["size"] > 0
    )
    if same:
        state["price"] = price
        state["size"] = min(state["size"], float(size))
        return
    touch = book_view["up_bid1"] if side == "bid" else book_view["up_ask1"]
    touch_size = book_view["up_bid1_size"] if side == "bid" else book_view["up_ask1_size"]
    queue = queue_ahead(price, touch, touch_size, side)
    shown = level_size(book_view, price, side)
    state.update(price=price, size=float(size), queue=queue, prev=shown, traded=0.0)


def _on_book(state, rec, side):
    if state["price"] is None:
        return
    new_size = level_size(rec, state["price"], side)
    queue, prev, traded = apply_cancel(state["queue"], state["prev"], new_size, state["traded"])
    state["queue"] = queue
    state["prev"] = prev
    state["traded"] = traded


def _usable_trades(trades):
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["ts", "up_side", "up_price", "size"])
    frame = trades
    if "paired" in frame.columns:
        frame = frame.loc[~frame["paired"].fillna(False).astype(bool)]
    frame = frame.loc[frame["up_side"].isin(["buy", "sell"]) & (frame["size"] > 0)]
    if frame.empty:
        return frame
    return frame[["ts", "up_side", "up_price", "size"]].sort_values("ts")


def _groups(frame, time_col):
    if frame is None or len(frame) == 0 or time_col not in frame.columns:
        return {}
    ordered = frame.sort_values(time_col)
    buckets = np.floor(ordered[time_col].to_numpy(dtype=float) - 1e-9).astype(np.int64)
    grouped = {}
    for bucket, row in zip(buckets, ordered.itertuples(index=False)):
        grouped.setdefault(int(bucket), []).append(row)
    return grouped


def _blank_order():
    return {"price": None, "size": 0.0, "queue": 0.0, "prev": None, "traded": 0.0}


def _by_slug(frame):
    if frame is None or len(frame) == 0:
        return {}
    return {slug: group for slug, group in frame.groupby("slug", sort=False)}


def _concat(frames):
    frames = [frame for frame in frames if frame is not None and len(frame)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _mid(rec):
    bid = getattr(rec, "up_bid1", None)
    ask = getattr(rec, "up_ask1", None)
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bid) or not math.isfinite(ask) or bid < 0 or ask < 0:
        return None
    return (bid + ask) / 2.0


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
