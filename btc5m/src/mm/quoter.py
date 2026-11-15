"""Inventory-skewed quoter. Quotes are passive against the book they are shown."""
from __future__ import annotations

import math

from common import to_tick


def _finite(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _floor_tick(price):
    return math.floor(price * 100.0 + 1e-9) / 100.0


def _ceil_tick(price):
    return math.ceil(price * 100.0 - 1e-9) / 100.0


class Quoter:
    def __init__(self, h_min, kappa, gamma, size, inv_cap, tau_stop, delta_s):
        self.h_min = float(h_min)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.size = float(size)
        self.inv_cap = float(inv_cap)
        self.tau_stop = float(tau_stop)
        self.delta_s = float(delta_s)

    def quote(self, p, sigma, tau, inventory, book):
        """Return (bid, ask, bid_size, ask_size).

        A missing price means that side is not quoted. `book` is the book the
        decision is allowed to see, plus optional `S` and `dp_dS` used in the
        half-spread. Models that ignore spot leave dp_dS at 0, so h = h_min.

        c = p - gamma * inventory
        h = max(h_min, kappa * |dp/dS| * S * sigma * sqrt(delta))
        """
        if not math.isfinite(_finite(p, default=float("nan"))) or _finite(tau, default=-1) < self.tau_stop:
            return None, None, 0.0, 0.0

        p = float(p)
        tau = float(tau)
        inventory = _finite(inventory)
        sigma = max(0.0, _finite(sigma))
        spot = max(0.0, _finite(book.get("S")))
        slope = abs(_finite(book.get("dp_dS")))
        horizon = math.sqrt(max(self.delta_s, 0.0))
        half = max(self.h_min, self.kappa * slope * spot * sigma * horizon)
        center = p - self.gamma * inventory

        bid = _floor_tick(center - half)
        ask = _ceil_tick(center + half)
        bid = min(0.99, max(0.01, bid))
        ask = min(0.99, max(0.01, ask))

        recorded_ask = to_tick(book.get("up_ask1"))
        recorded_bid = to_tick(book.get("up_bid1"))
        if recorded_ask is not None:
            bid = min(bid, round(recorded_ask - 0.01, 2))
        if recorded_bid is not None:
            ask = max(ask, round(recorded_bid + 0.01, 2))

        if bid < 0.01:
            bid = None
        if ask > 0.99:
            ask = None
        if bid is not None and ask is not None and bid >= ask - 1e-12:
            return None, None, 0.0, 0.0

        bid_size = min(self.size, self.inv_cap - inventory) if bid is not None else 0.0
        ask_size = min(self.size, self.inv_cap + inventory) if ask is not None else 0.0
        if bid_size <= 1e-9:
            bid, bid_size = None, 0.0
        if ask_size <= 1e-9:
            ask, ask_size = None, 0.0
        if bid is None and ask is None:
            return None, None, 0.0, 0.0
        return bid, ask, float(bid_size), float(ask_size)
