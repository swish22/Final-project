"""Unit checks for the digital pricer, the quoter, queue fills, the replay and the causal grid."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import assign_split, block_bootstrap
from grid import Spot, attach_spot, basis_table, ewma_sigma, fit_basis, market_pm_grid, market_spot_table
from mm.queue_fill import INF_QUEUE, apply_trade, end_of_second, queue_ahead, visible_size
from mm.quoter import Quoter
from mm.replay import bucket_trades, replay_market
from mm.taker import Panel, max_position, simulate
from models.d1 import d1_components


# ---------------------------------------------------------------- digital model

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


def test_d1_is_unpriced_without_volatility():
    p, _, _ = d1_components([64_010.0], [0.0], [64_000.0], [np.nan], [100.0])
    assert np.isnan(p[0])


def test_basis_noise_keeps_d1_uncertain_near_the_strike_at_expiry():
    # One dollar above the strike with no time left: certain without noise, a coin flip with it.
    plain, _, _ = d1_components([64_001.0], [0.0], [64_000.0], [1e-4], [0.0])
    noisy, _, _ = d1_components([64_001.0], [0.0], [64_000.0], [1e-4], [0.0], basis_sd=3.0)
    assert plain[0] == 0.995
    assert 0.5 < noisy[0] < 0.7


# ---------------------------------------------------------------- quoter

def test_quoter_never_locks_or_crosses_the_book():
    quoter = Quoter(h_min=0.01, kappa=2.0, gamma=0.001, size=10, inv_cap=100, tau_stop=5, delta_s=1)
    books = [(0.40, 0.42), (0.50, 0.51), (0.10, 0.90), (0.48, 0.49), (0.55, 0.50), (0.985, 0.997), (0.003, 0.012)]
    for p in np.linspace(0.02, 0.98, 25):
        for bid_px, ask_px in books:
            bid, ask, _, _ = quoter.quote(
                p, sigma=1e-4, tau=100, inventory=0,
                book={"up_bid1": bid_px, "up_ask1": ask_px, "S": 64_000.0, "dp_dS": 1e-4},
            )
            if bid is not None and ask is not None:
                assert bid < ask
            if bid is not None:
                assert bid < ask_px - 1e-9
            if ask is not None:
                assert ask > bid_px + 1e-9


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


def test_quoter_skews_against_inventory_and_widens_with_volatility():
    book = {"up_bid1": 0.10, "up_ask1": 0.90, "S": 64_000.0, "dp_dS": 2e-3}
    flat = Quoter(h_min=0.02, kappa=0.0, gamma=0.002, size=10, inv_cap=100, tau_stop=0, delta_s=1)
    bid0, ask0, _, _ = flat.quote(0.5, 1e-4, 100, 0, book)
    bid1, ask1, _, _ = flat.quote(0.5, 1e-4, 100, 50, book)
    assert (bid0, ask0) == (0.48, 0.52)
    assert (bid1, ask1) == (0.38, 0.42)
    # kappa * |dp/dS| * S * sigma * sqrt(delta + 1) = 4 * 2e-3 * 64000 * 1e-4 * sqrt(1) = 0.0512
    wide = Quoter(h_min=0.02, kappa=4.0, gamma=0.0, size=10, inv_cap=100, tau_stop=0, delta_s=0)
    bid, ask, _, _ = wide.quote(0.5, 1e-4, 100, 0, book)
    assert (bid, ask) == (0.44, 0.56)
    # Three more seconds of latency doubles the horizon's square root, and the half-spread with it.
    slow = Quoter(h_min=0.02, kappa=4.0, gamma=0.0, size=10, inv_cap=100, tau_stop=0, delta_s=3)
    bid, ask, _, _ = slow.quote(0.5, 1e-4, 100, 0, book)
    assert (bid, ask) == (0.39, 0.61)


# ---------------------------------------------------------------- queue rules (prices in tenths of a cent)

def test_queue_ahead_by_position_in_the_book():
    levels = (500, 100.0, 480, 40.0)
    assert queue_ahead(510, levels, "bid") == 0          # better than the touch
    assert queue_ahead(500, levels, "bid") == 100        # joins the touch
    assert queue_ahead(490, levels, "bid") == 0          # empty tick between the two levels
    assert queue_ahead(480, levels, "bid") == 40         # joins the second level
    assert queue_ahead(470, levels, "bid") == INF_QUEUE  # deeper than the recording
    assert visible_size(470, levels, "bid") is None
    asks = (510, 80.0, 520, 30.0)
    assert queue_ahead(500, asks, "ask") == 0
    assert queue_ahead(520, asks, "ask") == 30
    assert queue_ahead(530, asks, "ask") == INF_QUEUE
    assert queue_ahead(500, (-1, 0.0, -1, 0.0), "bid") == 0


def test_behind_the_recording_fills_only_on_a_through_trade():
    queue, size, fill = apply_trade(INF_QUEUE, 10, 500, 20, 480, "bid")
    assert fill == 0 and size == 10
    queue, size, fill = apply_trade(INF_QUEUE, 10, 480, 20, 480, "bid")
    assert fill == 0 and size == 10
    queue, size, fill = apply_trade(INF_QUEUE, 10, 470, 4, 480, "bid")
    assert fill == 4 and size == 6


def test_join_does_not_fill_until_the_queue_is_gone():
    queue, size, fill = apply_trade(100, 10, 500, 40, 500, "bid")
    assert fill == 0 and abs(queue - 60) < 1e-9
    queue, size, fill = apply_trade(queue, size, 500, 60, 500, "bid")
    assert fill == 0 and queue == 0
    queue, size, fill = apply_trade(queue, size, 500, 5, 500, "bid")
    assert fill == 5 and size == 5


def test_end_of_second_cautious_and_generous():
    # 100 displayed when we joined, 70 a second later, 10 of that traded.
    assert end_of_second(90, 100, 70, 10, "cautious") == 70      # only the impossible part comes off
    assert end_of_second(90, 100, 70, 10, "generous") == 70      # 20 cancelled ahead: 90 - 20
    assert end_of_second(50, 100, 70, 10, "cautious") == 50      # cancels assumed behind us
    assert end_of_second(50, 100, 70, 10, "generous") == 30      # cancels assumed ahead of us
    assert end_of_second(50, 100, 70, 30, "generous") == 50      # the shrink was all trades
    assert end_of_second(50, 100, None, 0, "generous") == 50     # level out of view: no information
    assert end_of_second(INF_QUEUE, None, 25, 0, "cautious") == 25


# ---------------------------------------------------------------- replay

def _grid(seconds, bids, asks, p=0.55, y=1, hi_bid=None, lo_ask=None, bid_size=100.0, ask_size=80.0):
    n = len(seconds)
    return pd.DataFrame({
        "slug": "m", "t": seconds, "tau": [seconds[-1] - t for t in seconds], "y": y,
        "p": p, "dp_dS": 0.0, "sigma": 1e-4, "S": 100.0,
        "up_bid1": bids, "up_bid1_size": bid_size, "up_bid2": np.nan, "up_bid2_size": np.nan,
        "up_ask1": asks, "up_ask1_size": ask_size, "up_ask2": np.nan, "up_ask2_size": np.nan,
        "hi_bid_next": hi_bid if hi_bid is not None else [np.nan] * n,
        "lo_ask_next": lo_ask if lo_ask is not None else [np.nan] * n,
        "pm_mid_filled": [(b + a) / 2 for b, a in zip(bids, asks)],
    })


def _trades(rows):
    return pd.DataFrame(rows, columns=["ts", "up_side", "up_price", "size"]).assign(paired=False)


def _quoter(h_min=0.05, tau_stop=0):
    return Quoter(h_min=h_min, kappa=0.0, gamma=0.0, size=10, inv_cap=100, tau_stop=tau_stop, delta_s=1)


def test_replay_join_fills_only_the_excess_over_the_queue():
    seconds = list(range(1000, 1004))
    grid = _grid(seconds, [0.50] * 4, [0.60] * 4)
    trades = _trades([(1000.5, "sell", 0.50, 150.0)])
    summary, fills, _ = replay_market(grid, trades, _quoter(), "cautious", delta_s=0)
    assert len(fills) == 1
    assert fills.iloc[0]["side"] == "buy" and fills.iloc[0]["kind"] == "trade"
    assert fills.iloc[0]["size"] == 10
    assert close(fills.iloc[0]["price"], 0.50)
    assert close(summary["pnl"], -5.0 + 10.0)


def test_replay_keeps_its_queue_place_while_the_price_is_unchanged():
    seconds = list(range(1000, 1004))
    grid = _grid(seconds, [0.50] * 4, [0.60] * 4)
    # 60 trades in the first second, 60 in the second: the 100 ahead of us is gone after 120.
    trades = _trades([(1000.5, "sell", 0.50, 60.0), (1001.5, "sell", 0.50, 60.0)])
    _, fills, _ = replay_market(grid, trades, _quoter(), "cautious", delta_s=0)
    assert len(fills) == 1 and fills.iloc[0]["t"] == 1001 and fills.iloc[0]["size"] == 10


def test_replay_rejects_a_quote_that_would_cross_the_live_book():
    # The decision saw an ask of 0.60 one second ago, but the ask is 0.50 when the bid arrives.
    seconds = list(range(1000, 1005))
    grid = _grid(seconds, [0.45, 0.45, 0.40, 0.40, 0.40], [0.60, 0.60, 0.50, 0.50, 0.50], p=0.57)
    trades = _trades([(1002.5, "sell", 0.40, 50.0)])
    summary, fills, _ = replay_market(grid, trades, _quoter(h_min=0.05), "cautious", delta_s=1)
    # At t = 1002 the bid computed from the 1001 row is 0.52 >= the live ask 0.50: rejected, not filled.
    assert not ((fills["t"] == 1002) & (fills["side"] == "buy")).any()
    assert summary["rejects"] >= 1


def test_replay_fills_a_resting_bid_when_the_ask_trades_through_it():
    seconds = list(range(1000, 1004))
    lo_ask = [np.nan, 0.50, np.nan, np.nan]
    grid = _grid(seconds, [0.50, 0.50, 0.44, 0.44], [0.60, 0.60, 0.45, 0.45], lo_ask=lo_ask, y=0)
    summary, fills, _ = replay_market(grid, _trades([]), _quoter(), "cautious", delta_s=0)
    hit = fills.loc[fills["kind"] == "cross"]
    assert len(hit) == 1 and hit.iloc[0]["t"] == 1001 and hit.iloc[0]["side"] == "buy"
    assert close(hit.iloc[0]["price"], 0.50) and hit.iloc[0]["size"] == 10
    assert summary["n_cross"] == 1


def test_replay_pnl_equals_the_sum_of_fill_profits_at_settlement():
    rng = np.random.default_rng(0)
    seconds = list(range(1000, 1061))
    mid = np.clip(0.5 + np.cumsum(rng.normal(0, 0.01, len(seconds))), 0.05, 0.95)
    bids, asks = np.round(mid - 0.005, 2), np.round(mid - 0.005, 2) + 0.01
    grid = _grid(seconds, bids, asks, p=np.round(mid, 3), y=1,
                 hi_bid=np.r_[bids[1:], np.nan], lo_ask=np.r_[asks[1:], np.nan])
    rows = [(t + rng.random(), rng.choice(["buy", "sell"]), float(rng.choice([b, a])), float(rng.integers(1, 200)))
            for t, b, a in zip(seconds[:-1], bids, asks) for _ in range(3)]
    for rule in ("cautious", "generous"):
        summary, fills, path = replay_market(grid, _trades(rows), _quoter(h_min=0.01), rule, delta_s=1)
        assert len(fills) > 5
        sign = np.where(fills["side"] == "buy", 1.0, -1.0)
        assert close(summary["pnl"], float((sign * (1.0 - fills["price"]) * fills["size"]).sum()))
        assert close(summary["final_inv"], float((sign * fills["size"]).sum()))
        assert abs(fills["inventory"]).max() <= 100 + 1e-9
        assert len(path) == len(seconds) - 1


def test_a_trade_is_assigned_to_the_second_it_can_fill():
    trades = _trades([(1000.0, "sell", 0.5, 1.0), (1000.2, "sell", 0.5, 2.0), (1001.0, "sell", 0.5, 3.0)])
    grouped = bucket_trades(trades, 1000, 5)
    # (t, t + 1]: the print at exactly 1000.0 came before the quote posted at 1000.
    assert sorted(grouped) == [0]
    assert [row[3] for row in grouped[0]] == [2.0, 3.0]


def test_paired_down_legs_are_not_counted_twice():
    trades = _trades([(1000.5, "sell", 0.5, 4.0), (1000.5, "sell", 0.5, 4.0)])
    trades.loc[1, "paired"] = True
    assert [row[3] for row in bucket_trades(trades, 1000, 5)[0]] == [4.0]


# ---------------------------------------------------------------- causal grid

def test_sigma_does_not_use_a_future_return():
    flat = np.array([100.0] * 20)
    jumped = flat.copy()
    jumped[-1] = 200.0
    assert np.allclose(ewma_sigma(flat, 5)[:-1], ewma_sigma(jumped, 5)[:-1], equal_nan=True)


def test_sigma_of_a_constant_volatility_path_is_that_volatility():
    prices = 100.0 * np.exp(np.cumsum(np.tile([0.001, -0.001], 200)))
    assert abs(ewma_sigma(prices, 30)[-1] - 0.001) < 1e-9
    assert abs(ewma_sigma(prices, 30)[3] - 0.001) < 1e-9


def _book(levels):
    rows = []
    for ts, bid, ask in levels:
        row = {"ts": ts}
        for name in ("up_bid1", "up_bid2", "up_ask1", "up_ask2"):
            row[name] = -1.0
            row[name + "_size"] = 0.0
        row.update(up_bid1=bid, up_bid1_size=100.0, up_ask1=ask, up_ask1_size=80.0)
        rows.append(row)
    return pd.DataFrame(rows)


MARKET = {"slug": "m", "start_unix": 1000, "end_unix": 1010}


def test_grid_row_ignores_a_later_book_update_and_trade():
    book = _book([(950.0, 0.40, 0.60), (1005.5, 0.70, 0.80)])
    trades = _trades([(1006.2, "buy", 0.77, 1.0)])
    full, _ = market_pm_grid(MARKET, book, trades)
    trimmed, _ = market_pm_grid(MARKET, book.iloc[:1], trades.iloc[:0])
    for col in ("up_bid1", "pm_mid", "pm_flow_30"):
        assert close(full.loc[full["t"] == 1005, col].iloc[0], trimmed.loc[trimmed["t"] == 1005, col].iloc[0])
    assert pd.isna(full.loc[full["t"] == 1006, "last_trade"].iloc[0])
    # The later update is visible once it is in the past, so the check above is not vacuous.
    assert close(full.loc[full["t"] == 1006, "up_bid1"].iloc[0], 0.70)
    assert close(full.loc[full["t"] == 1007, "last_trade"].iloc[0], 0.77)
    assert close(full.loc[full["t"] == 1007, "pm_flow_30"].iloc[0], 1.0)
    # The extremes of the next second are simulator inputs: 1005 sees the 0.80 ask arrive in (1005, 1006].
    assert close(full.loc[full["t"] == 1005, "lo_ask_next"].iloc[0], 0.80)


def test_one_sided_book_still_has_a_mid_and_keeps_sub_cent_prices():
    book = _book([(950.0, 0.997, -1.0)])
    grid, stats = market_pm_grid(MARKET, book, _trades([]))
    row = grid.iloc[0]
    assert close(row["up_bid1"], 0.997) and pd.isna(row["up_ask1"])
    assert pd.isna(row["pm_mid"]) and close(row["pm_mid_filled"], 0.9985)
    assert stats["no_book_frac"] == 0.0 and stats["one_sided_frac"] == 1.0


def _spot_fixture():
    secs = np.arange(900, 1311)
    price = 64_000.0 + (secs - 1000) * 0.5
    frame = pd.DataFrame({"sec": secs, "binance_mid": price + 40.0, "coinbase_mid": price,
                          "binance_signed_vol": 0.0, "binance_signed_vol_10s": 0.0, "seg": 1})
    markets = pd.DataFrame([{
        "slug": "m", "start_unix": 1000, "end_unix": 1300, "day_idx": 0, "K": 64_003.0,
        "open_ts": 1015.2, "winner": "up", "res_close": 64_150.0,
    }])
    return Spot(frame), markets


def test_the_strike_is_not_used_before_it_was_recorded():
    spot, markets = _spot_fixture()
    table = market_spot_table(markets, spot)
    basis = basis_table(markets, table, "coinbase")
    pm = pd.DataFrame({"slug": "m", "t": np.arange(1000, 1301)})
    g = attach_spot(pm, markets.merge(table, on="slug"), spot, basis, "coinbase", 60).set_index("t")
    assert close(g.loc[1015, "K"], 64_000.0) and close(g.loc[1015, "b"], 0.0)   # still the proxy at the open
    assert close(g.loc[1016, "K"], 64_003.0) and close(g.loc[1016, "b"], 3.0)   # strike known from 1015.2
    assert close(g.loc[1000, "S_open"], 64_000.0) and g.loc[1000, "tau"] == 300
    assert g["y"].eq(1).all()


def test_basis_fit_recovers_persistence_and_noise():
    rng = np.random.default_rng(1)
    n = 4000
    prior = np.full(n, -20.0)
    open_dev = rng.normal(0, 4, n)
    noise = rng.normal(0, 2, n)
    basis = pd.DataFrame({"slug": [f"m{i}" for i in range(n)], "basis_prior": prior,
                          "basis_open": prior + open_dev, "basis_close": prior + 0.3 * open_dev + noise})
    fit = fit_basis(basis, basis["slug"])
    assert abs(fit["beta"] - 0.3) < 0.03 and abs(fit["sd"] - 2.0) < 0.15


# ---------------------------------------------------------------- taker strategy

def _taker_panel(model, bid, ask, strike=None, y=1, end=1020):
    n = len(model)
    t = np.arange(end - n + 1, end + 1)
    return Panel(pd.DataFrame({
        "slug": "m", "t": t, "tau": end - t, "y": y, "split": "test", "day_idx": 0, "start_unix": int(t[0]),
        "up_bid1": bid, "up_bid1_size": 50.0, "up_ask1": ask, "up_ask1_size": 4.0,
        "pm_mid_filled": (np.asarray(bid) + np.asarray(ask)) / 2, "p_D1": model,
        "K": 100.0 if strike is None else strike, "b": 0.0,
    }))


def test_taker_buys_the_stale_ask_and_closes_at_the_later_bid():
    # The model jumps 0.10 at index 5; the book has not moved then, and is 0.08 higher three seconds later.
    model = np.r_[np.full(5, 0.50), np.full(16, 0.60)]
    bid = np.r_[np.full(7, 0.49), np.full(14, 0.57)]
    ask = bid + 0.01
    trades = simulate(_taker_panel(model, bid, ask), theta=0.05, follow_max=0.5, hold_s=3, tau_min=5)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["second"] == 5 and row["side"] == 1
    assert close(row["price_in"], 0.50) and close(row["price_out"], 0.57)
    assert close(row["qty"], 4.0)                       # only the displayed size is taken
    assert close(row["pnl"], (0.57 - 0.50) * 4.0)


def test_taker_stands_aside_when_polymarket_has_already_followed():
    model = np.r_[np.full(5, 0.50), np.full(16, 0.60)]
    bid = np.r_[np.full(5, 0.49), np.full(16, 0.57)]    # the book moved in the same second
    trades = simulate(_taker_panel(model, bid, bid + 0.01), theta=0.05, follow_max=0.5, hold_s=3, tau_min=5)
    assert len(trades) == 0
    # With no limit on how far the market may have followed, the same jump is traded.
    trades = simulate(_taker_panel(model, bid, bid + 0.01), theta=0.05, follow_max=np.inf, hold_s=3, tau_min=5)
    assert len(trades) == 1


def test_taker_ignores_the_jump_when_the_strike_is_first_recorded():
    model = np.r_[np.full(5, 0.50), np.full(16, 0.60)]
    bid = np.full(21, 0.49)
    strike = np.r_[np.full(5, 100.0), np.full(16, 101.0)]   # the strike changes in the jump second
    trades = simulate(_taker_panel(model, bid, bid + 0.01, strike), theta=0.05, follow_max=0.5, hold_s=3, tau_min=5)
    assert len(trades) == 0


def test_taker_delay_fee_and_settlement():
    model = np.r_[np.full(5, 0.50), np.full(16, 0.40)]      # a fall: sell at the bid
    bid = np.r_[np.full(6, 0.49), np.full(15, 0.41)]
    ask = bid + 0.01
    panel = _taker_panel(model, bid, ask, y=0)
    now = simulate(panel, theta=0.05, follow_max=0.5, hold_s=3, tau_min=5).iloc[0]
    assert now["side"] == -1 and close(now["price_in"], 0.49) and close(now["price_out"], 0.42)
    late = simulate(panel, theta=0.05, follow_max=0.5, hold_s=3, tau_min=5, delay_s=1).iloc[0]
    assert close(late["price_in"], 0.41) and close(late["pnl"], (0.41 - 0.42) * 10.0)   # sold after the move
    paid = simulate(panel, theta=0.05, follow_max=0.5, hold_s=3, tau_min=5, fee_rate=0.07).iloc[0]
    fee = 0.07 * (0.49 * 0.51 + 0.42 * 0.58) * 10.0
    assert close(paid["pnl"], now["pnl"] - fee)
    # Held past the end of the market, the trade settles at the outcome (Down: the Up token is worth 0).
    held = simulate(panel, theta=0.05, follow_max=0.5, hold_s=30, tau_min=5).iloc[0]
    assert bool(held["settled"]) and close(held["price_out"], 0.0) and close(held["pnl"], 0.49 * 10.0)
    assert max_position(panel, simulate(panel, theta=0.05, follow_max=0.5, hold_s=3, tau_min=5), 3) == 10.0


# ---------------------------------------------------------------- helpers

def test_split_by_day_index_ranges():
    out = assign_split([0, 10, 11, 13, 14, 16, 17], {"train": [0, 10], "val": [11, 13], "test": [14, 16]})
    assert list(out) == ["train", "train", "val", "val", "test", "test", None]


def test_block_bootstrap_interval_covers_the_mean():
    rng = np.random.default_rng(2)
    values = rng.normal(1.0, 2.0, 600)
    mean, lo, hi, se = block_bootstrap(values, np.ones(600), np.arange(600) // 12)
    assert close(mean, values.mean()) and lo < mean < hi
    assert 0.05 < se < 0.12


def close(actual, expected, tol=1e-6):
    return abs(float(actual) - float(expected)) < tol
