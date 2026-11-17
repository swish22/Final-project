"""Inventory-skewed quoter. Quotes are passive against the book they are shown."""
from __future__ import annotations

import math

from common import mills

NONE = -1
REFRESH_S = 1.0


class Quoter:
    def __init__(self, h_min, kappa, gamma, size, inv_cap, tau_stop, delta_s):
        self.h_min = float(h_min)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.size = float(size)
        self.inv_cap = float(inv_cap)
        self.tau_stop = float(tau_stop)
        self.delta_s = float(delta_s)
        # A quote is built from data delta seconds old and then stays live for one second,
        # so the move it has to survive is over delta + 1 seconds.
        self.sqrt_horizon = math.sqrt(max(self.delta_s, 0.0) + REFRESH_S)

    def quote(self, p, sigma, tau, inventory, book):
        """Return (bid, ask, bid_size, ask_size) in dollars.

        A missing price means that side is not quoted. `book` is the book the
        decision is allowed to see, plus optional `S` and `dp_dS` used in the
        half-spread. Models that ignore spot leave dp_dS at 0, so h = h_min.

        c = p - gamma * inventory
        h = max(h_min, kappa * |dp/dS| * S * sigma * sqrt(delta + 1))
        """
        bid, ask, bid_size, ask_size = self.quote_mills(
            _number(p), _number(sigma), _number(tau), _number(inventory, 0.0),
            _number(book.get("S")), _number(book.get("dp_dS")),
            mills(book.get("up_bid1")), mills(book.get("up_ask1")),
        )
        return (
            None if bid == NONE else bid / 1000.0,
            None if ask == NONE else ask / 1000.0,
            bid_size, ask_size,
        )

    def quote_mills(self, p, sigma, tau, inventory, spot, slope, seen_bid, seen_ask):
        """Same rule with prices in tenths of a cent and -1 for a missing price.

        Our quotes sit on whole cents, which is a valid tick everywhere, and stay
        at least one cent away from the far side of the book the decision saw.
        """
        if p != p or tau != tau or tau < self.tau_stop:
            return NONE, NONE, 0.0, 0.0
        half = self.h_min
        if self.kappa > 0.0:
            wide = self.kappa * abs(slope) * spot * sigma * self.sqrt_horizon
            if wide == wide and wide > half:
                half = wide
        center = p - self.gamma * inventory
        bid = int(math.floor((center - half) * 100.0 + 1e-9)) * 10
        ask = int(math.ceil((center + half) * 100.0 - 1e-9)) * 10
        bid = min(990, max(10, bid))
        ask = min(990, max(10, ask))
        if seen_ask != NONE:
            # Largest whole cent strictly below the ask.
            bid = min(bid, (seen_ask - 1) // 10 * 10)
        if seen_bid != NONE:
            ask = max(ask, (seen_bid // 10 + 1) * 10)
        if bid < 10:
            bid = NONE
        if ask > 990:
            ask = NONE
        if bid != NONE and ask != NONE and bid >= ask:
            return NONE, NONE, 0.0, 0.0

        bid_size = min(self.size, self.inv_cap - inventory) if bid != NONE else 0.0
        ask_size = min(self.size, self.inv_cap + inventory) if ask != NONE else 0.0
        if bid_size <= 1e-9:
            bid, bid_size = NONE, 0.0
        if ask_size <= 1e-9:
            ask, ask_size = NONE, 0.0
        return bid, ask, float(bid_size), float(ask_size)


def _number(value, default=float("nan")):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default
