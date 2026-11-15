"""Queue position for one resting order.

Better than the touch: queue ahead is 0, so the next taker at our price fills us.
Joining the touch: queue ahead starts at the displayed size.
Worse than the touch: queue ahead is infinite, and only a trade that prints strictly
through our price can fill us.

Cautious mode lets only trades reduce a finite queue. Generous mode also treats
shrinks in displayed size, beyond the trades already counted, as cancels ahead of us.
"""
from __future__ import annotations

import math

from common import cents

INF_QUEUE = math.inf


def queue_ahead(our_price, touch_price, touch_size, side):
    ours = cents(our_price)
    touch = cents(touch_price)
    if ours is None:
        return 0.0
    better = touch is None or (side == "bid" and ours > touch) or (side == "ask" and ours < touch)
    if better:
        return 0.0
    if ours == touch:
        try:
            size = float(touch_size)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(size) or size <= 0:
            return 0.0
        return size
    return INF_QUEUE


def apply_trade(queue, order_size, trade_price, trade_size, our_price, side):
    """Apply one taker print. Returns (queue, remaining size, fill size)."""
    if our_price is None or order_size <= 1e-12 or trade_size <= 0:
        return queue, order_size, 0.0
    ours = cents(our_price)
    trade = cents(trade_price)
    if ours is None or trade is None:
        return queue, order_size, 0.0
    if side == "bid":
        if trade > ours:
            return queue, order_size, 0.0
        through = trade < ours
    else:
        if trade < ours:
            return queue, order_size, 0.0
        through = trade > ours

    trade_size = float(trade_size)
    if through:
        fill = min(order_size, trade_size)
        return 0.0, _dust(order_size - fill), fill

    if not math.isfinite(queue):
        return queue, order_size, 0.0
    if queue >= trade_size - 1e-12:
        return _dust(queue - trade_size), order_size, 0.0
    fill = min(order_size, trade_size - queue)
    return 0.0, _dust(order_size - fill), fill


def apply_cancel(queue, prev_displayed, new_displayed, traded_since):
    """Generous shrink: displayed-size drop minus trades already applied.

    Returns (queue, displayed size to remember, traded-since reset to 0).
    A missing new size leaves the queue unchanged.
    """
    if new_displayed is None or not math.isfinite(queue):
        remembered = prev_displayed if new_displayed is None else float(new_displayed)
        traded = traded_since if new_displayed is None else 0.0
        return queue, remembered, traded
    shown = max(0.0, float(new_displayed))
    if prev_displayed is not None:
        cancel = (float(prev_displayed) - shown) - float(traded_since)
        if cancel > 1e-9:
            queue = max(0.0, queue - cancel)
    return _dust(queue), shown, 0.0


def level_size(book, price, side):
    """Displayed size at `price` on the top two levels, or None if it is not shown."""
    tick = cents(price)
    if tick is None or book is None:
        return None
    if side == "bid":
        pairs = (("up_bid1", "up_bid1_size"), ("up_bid2", "up_bid2_size"))
    else:
        pairs = (("up_ask1", "up_ask1_size"), ("up_ask2", "up_ask2_size"))
    for price_key, size_key in pairs:
        if cents(_lookup(book, price_key)) != tick:
            continue
        try:
            size = float(_lookup(book, size_key))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(size) or size < 0:
            return 0.0
        return size
    return None


def _lookup(book, key):
    if isinstance(book, dict):
        return book.get(key)
    return getattr(book, key)


def _dust(value):
    if value < 1e-9:
        return 0.0
    return float(value)
