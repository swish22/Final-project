"""Unit checks for the digital pricer, the quoter, queue fills, and the causal grid."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid import build_market_grid, ewma_sigma
from mm.queue_fill import apply_cancel, apply_trade, queue_ahead
from mm.quoter import Quoter
from mm.replay import replay_market
from models.d1 import d1_components


def test_d1_is_one_half_at_the_open():
    # b = K - S_open, and S is still the open, so (S + b) / K = 1.
    p, dp, z = d1_components([64_000.0], [125.0], [64_125.0], [1e-4], [200.0])
    assert abs(p[0] - 0.5) < 1e-12
    assert abs(z[0]) < 1e-9
    # phi(0) / (K * sigma * sqrt(tau))
    expected = (1.0 / math.sqrt(2.0 * math.pi)) / (64_125.0 * 1e-4 * math.sqrt(200.0))
    assert abs(dp[0] - expected) / expected < 1e-6


def test_d1_clips_when_spot_is_through_the_strike():
    above, _, _ = d1_components([70_000.0], [0.0], [64_000.0], [1e-5], [1.0])
    below, _, _ = d1_components([10_000.0], [0.0], [64_000.0], [1e-4], [1.0])
    at_expiry, _, z = d1_components([70_000.0], [0.0], [64_000.0], [1e-4], [0.0])
    assert above[0] == 0.995
    assert below[0] == 0.005
    assert at_expiry[0] == 0.995
    assert z[0] == 8.0


def test_quoter_never_locks_or_crosses_the_book():
    quoter = Quoter(h_min=0.01, kappa=2.0, gamma=0.001, size=10, inv_cap=100, tau_stop=5, delta_s=1)
    books = [(0.40, 0.42), (0.50, 0.51), (0.10, 0.90), (0.48, 0.49), (0.55, 0.50)]
    for p in np.linspace(0.02, 0.98, 25):
        for bid_px, ask_px in books:
            bid, ask, _, _ = quoter.quote(
                p, sigma=1e-4, tau=100, inventory=0,
                book={"up_bid1": bid_px, "up_ask1": ask_px, "S": 64_000.0, "dp_dS": 1e-4},
            )
            if bid is not None and ask is not None:
                assert bid < ask
            if bid is not None and ask_px > 0:
                assert bid <= round(ask_px - 0.01, 2) + 1e-9
            if ask is not None and bid_px > 0:
                assert ask >= round(bid_px + 0.01, 2) - 1e-9


def test_quoter_stops_and_respects_the_inventory_cap():
    quoter = Quoter(h_min=0.01, kappa=0.0, gamma=0.0, size=10, inv_cap=100, tau_stop=15, delta_s=1)
    book = {"up_bid1": 0.40, "up_ask1": 0.60, "S": 64_000.0, "dp_dS": 0.0}
    assert quoter.quote(0.5, 1e-4, tau=14, inventory=0, book=book)[0] is None
    bid, ask, bid_size, ask_size = quoter.quote(0.5, 1e-4, tau=100, inventory=100, book=book)
    assert bid is None and bid_size == 0
    assert ask is not None and ask_size == 10
    bid, ask, bid_size, ask_size = quoter.quote(0.5, 1e-4, tau=100, inventory=-100, book=book)
    assert ask is None and ask_size == 0
    assert bid is not None and bid_size == 10


def test_behind_touch_fills_only_on_a_through_trade():
    assert queue_ahead(0.48, 0.50, 100, "bid") == math.inf
    queue, size, fill = apply_trade(math.inf, 10, 0.50, 20, 0.48, "bid")
    assert fill == 0 and size == 10
    queue, size, fill = apply_trade(math.inf, 10, 0.47, 4, 0.48, "bid")
    assert fill == 4 and size == 6


def test_join_does_not_fill_until_the_queue_is_gone():
    assert queue_ahead(0.50, 0.50, 100, "bid") == 100
    queue, size, fill = apply_trade(100, 10, 0.50, 40, 0.50, "bid")
    assert fill == 0 and abs(queue - 60) < 1e-9
    queue, size, fill = apply_trade(queue, size, 0.50, 60, 0.50, "bid")
    assert fill == 0 and queue == 0
    queue, size, fill = apply_trade(queue, size, 0.50, 5, 0.50, "bid")
    assert fill == 5 and size == 5


def test_generous_cancels_do_not_double_count_trades():
    # A shrink of 60 leaves 40 shares ahead. The drop itself is not the new queue.
    queue, _, _ = apply_cancel(100, prev_displayed=100, new_displayed=40, traded_since=0)
    assert abs(queue - 40) < 1e-9
    queue, _, _ = apply_cancel(80, prev_displayed=100, new_displayed=70, traded_since=30)
    assert abs(queue - 80) < 1e-9


def test_sigma_does_not_use_a_future_return():
    flat = np.array([100.0] * 20)
    jumped = flat.copy()
    jumped[-1] = 200.0
    assert np.allclose(ewma_sigma(flat, 5)[:-1], ewma_sigma(jumped, 5)[:-1], equal_nan=True)


def test_grid_row_ignores_a_later_book_update():
    market = pd.Series({
        "slug": "m", "start_unix": 1000, "end_unix": 1010, "K": 100.0, "winner": "up",
    })
    btc = pd.DataFrame({
        "sec": np.arange(900, 1011),
        "binance_mid": 100.0,
        "coinbase_mid": 100.0,
        "binance_signed_vol": 0.0,
    })
    book = _book([(950.0, 0.40, 0.60), (1005.5, 0.70, 0.80)])
    trades = pd.DataFrame({
        "ts": [1006.2], "up_price": [0.77], "up_side": ["buy"], "size": [1.0], "paired": [False],
    })
    full, _ = build_market_grid(market, btc, book, trades, half_life_s=30)
    trimmed, _ = build_market_grid(market, btc, book.iloc[:1].copy(), trades.iloc[:0].copy(), 30)
    at_1000 = full.loc[full["t"] == 1000].iloc[0]
    at_1000_trimmed = trimmed.loc[trimmed["t"] == 1000].iloc[0]
    assert close(at_1000["up_bid1"], 0.40)
    assert close(at_1000_trimmed["up_bid1"], 0.40)
    assert close(at_1000["pm_mid"], at_1000_trimmed["pm_mid"])
    assert pd.isna(at_1000["last_trade"]) and pd.isna(at_1000_trimmed["last_trade"])
    # The later update is visible once it is in the past, so the check above is not vacuous.
    assert close(full.loc[full["t"] == 1006, "up_bid1"].iloc[0], 0.70)
    assert close(full.loc[full["t"] == 1007, "last_trade"].iloc[0], 0.77)


def test_replay_join_fills_only_the_excess_over_the_queue():
    grid = pd.DataFrame([
        {
            "slug": "m", "t": t, "tau": 100, "end_unix": 1003, "y": 1, "split": "test",
            "p": 0.55, "dp_dS": 0.0, "sigma": 1e-4, "S": 100.0,
            "up_bid1": 0.50, "up_bid1_size": 100.0,
            "up_ask1": 0.60, "up_ask1_size": 80.0,
            "pm_mid": 0.55,
        }
        for t in range(1000, 1004)
    ])
    book = _book([(999.0, 0.50, 0.60)])
    trades = pd.DataFrame({
        "ts": [1000.5], "up_side": ["sell"], "up_price": [0.50], "size": [150.0], "paired": [False],
    })
    quoter = Quoter(h_min=0.05, kappa=0.0, gamma=0.0, size=10, inv_cap=100, tau_stop=0, delta_s=1)
    fills, path = replay_market(grid, book, trades, quoter, "cautious", delta_s=0)
    assert len(fills) == 1
    assert fills.iloc[0]["side"] == "buy"
    assert fills.iloc[0]["size"] == 10
    assert close(fills.iloc[0]["price"], 0.50)
    settled = path.loc[path["settled"], "equity"].iloc[0]
    assert close(settled, -5.0 + 10.0)


def _book(levels):
    rows = []
    for ts, bid, ask in levels:
        row = {"ts": ts}
        for name in ("up_bid1", "up_bid2", "up_ask1", "up_ask2"):
            row[name] = -1.0
            row[name + "_size"] = 0.0
        row["up_bid1"] = bid
        row["up_bid1_size"] = 100.0
        row["up_ask1"] = ask
        row["up_ask1_size"] = 80.0
        rows.append(row)
    return pd.DataFrame(rows)


def close(actual, expected, tol=1e-6):
    return abs(float(actual) - float(expected)) < tol
