"""Queue position for one resting order. Prices are integers in tenths of a cent.

`levels` is our side of the recorded book: (price1, size1, price2, size2), with -1
for a level that does not exist. Only two levels are recorded.

Queue ahead of a new order:
    better than the touch, or on an empty tick inside the two levels   0
    at a recorded level                                              its displayed size
    deeper than the second level                                     unknown (infinite)

A taker print at our price first eats the queue, then fills us. A print strictly
through our price fills us regardless of the queue, because the level was swept.

At the end of each second the displayed size at our price is read again:
    cautious   a shrink is assumed to be orders behind us, unless that is impossible
               (the queue can never exceed what is displayed)
    generous   a shrink that trades do not explain is assumed to be cancels ahead of us
"""
from __future__ import annotations

import math

INF_QUEUE = math.inf
NONE = -1


def visible_size(price, levels, side):
    """Displayed size at `price`: a size, 0.0 for an empty tick in view, None if out of view."""
    p1, s1, p2, s2 = levels
    if p1 == NONE:
        return 0.0
    if price == p1:
        return s1
    better = price > p1 if side == "bid" else price < p1
    if better:
        return 0.0
    if p2 == NONE:
        return 0.0
    if price == p2:
        return s2
    inside = price > p2 if side == "bid" else price < p2
    return 0.0 if inside else None


def queue_ahead(price, levels, side):
    shown = visible_size(price, levels, side)
    return INF_QUEUE if shown is None else shown


def apply_trade(queue, order_size, trade_price, trade_size, our_price, side):
    """Apply one taker print against our side. Returns (queue, remaining size, fill size)."""
    if our_price == NONE or order_size <= 1e-12 or trade_size <= 0:
        return queue, order_size, 0.0
    if side == "bid":
        if trade_price > our_price:
            return queue, order_size, 0.0
        through = trade_price < our_price
    else:
        if trade_price < our_price:
            return queue, order_size, 0.0
        through = trade_price > our_price

    if through:
        fill = min(order_size, trade_size)
        return 0.0, _dust(order_size - fill), fill
    if queue == INF_QUEUE:
        return queue, order_size, 0.0
    if queue >= trade_size - 1e-12:
        return _dust(queue - trade_size), order_size, 0.0
    fill = min(order_size, trade_size - queue)
    return 0.0, _dust(order_size - fill), fill


def end_of_second(queue, prev_shown, new_shown, traded, rule):
    """Update the queue from the displayed size one second later. Returns the new queue."""
    if new_shown is None:
        return queue
    if queue == INF_QUEUE:
        # The level came into view. Everything displayed is assumed to be ahead of us.
        return float(new_shown)
    if rule == "generous" and prev_shown is not None:
        cancel = (prev_shown - new_shown) - traded
        if cancel > 1e-9:
            queue = max(0.0, queue - cancel)
    return _dust(min(queue, new_shown))


def _dust(value):
    return 0.0 if value < 1e-9 else float(value)
