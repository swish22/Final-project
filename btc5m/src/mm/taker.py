"""A taker strategy: hit the stale quote when Bitcoin has moved and Polymarket has not.

Each second t:

    jump   = D1[t] - D1[t-1]      how far the BTC model's probability moved in the last second
    follow = mid[t] - mid[t-1]    how far the Polymarket mid moved in the same second

If jump >= theta and Polymarket has followed less than `follow_max` of it, the strategy
buys `size` shares at the displayed ask. The mirror case sells at the displayed bid.
Each trade is closed `hold_s` seconds later at the opposite touch, or at settlement if
the market ends first or that side of the book is empty.

Trades are only taken on a two-sided book, and never in the second where the strike
first becomes known: the model's probability jumps there because the model learned the
strike, not because Bitcoin moved.

Idealised on purpose: zero reaction time (the decision at t uses data up to t and fills
against the book displayed at t), no fees, and the displayed size is available.
`delay_s` and `fee_rate` exist to show what those assumptions are worth.

Every kept market has one grid row per second from its start to its end, so the
simulation runs on (markets x seconds) arrays with no loops over time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import HOUR_S, block_bootstrap

COLUMNS = ["slug", "t", "tau", "y", "split", "day_idx", "start_unix", "up_bid1", "up_bid1_size",
           "up_ask1", "up_ask1_size", "pm_mid_filled", "p_D1", "K", "b"]
TRADE_COLUMNS = ["market", "second", "side", "qty", "price_in", "price_out", "settled", "jump", "follow", "pnl"]


class Panel:
    """The grid as (markets x seconds) arrays."""

    def __init__(self, frame):
        frame = frame.sort_values(["slug", "t"])
        sizes = frame.groupby("slug", sort=False).size()
        width = int(sizes.iloc[0])
        if not (sizes == width).all():
            raise ValueError("every market needs the same number of grid rows")
        shape = (len(sizes), width)

        def grid(col):
            return frame[col].to_numpy(dtype=float).reshape(shape)

        first = frame.groupby("slug", sort=False).first()
        self.slug = first.index.to_numpy()
        self.split = first["split"].to_numpy()
        self.day = first["day_idx"].to_numpy()
        self.start = first["start_unix"].to_numpy()
        self.payoff = first["y"].to_numpy(dtype=float)
        self.bid, self.ask = grid("up_bid1"), grid("up_ask1")
        self.bid_size, self.ask_size = np.nan_to_num(grid("up_bid1_size")), np.nan_to_num(grid("up_ask1_size"))
        self.mid, self.model, self.tau = grid("pm_mid_filled"), grid("p_D1"), grid("tau")
        strike, basis = grid("K"), grid("b")
        # False in the second where the strike (or its basis) changed, i.e. when it was first recorded.
        self.same_strike = (strike == shift(strike, -1)) & (basis == shift(basis, -1))


def shift(a, k):
    """Value k columns to the right (k > 0: later in time), NaN past the edge."""
    if k == 0:
        return a.copy()
    out = np.full(a.shape, np.nan)
    if k > 0:
        out[:, :-k] = a[:, k:]
    else:
        out[:, -k:] = a[:, :k]
    return out


def signals(panel):
    """(jump, follow, tradable): the model's and the market's one-second moves, and where both are defined."""
    jump = panel.model - shift(panel.model, -1)
    follow = panel.mid - shift(panel.mid, -1)
    two_sided = np.isfinite(panel.bid) & np.isfinite(panel.ask)
    tradable = two_sided & (shift(two_sided.astype(float), -1) == 1.0) & panel.same_strike
    return jump, follow, tradable


def simulate(panel, theta, follow_max, hold_s, tau_min=10, size=10.0, delay_s=0, fee_rate=0.0):
    """Returns a dataframe with one row per trade."""
    jump, follow, tradable = signals(panel)
    # The decision is made at t; the order reaches the book delay_s seconds later.
    ask_in, bid_in = shift(panel.ask, delay_s), shift(panel.bid, delay_s)
    ask_room, bid_room = shift(panel.ask_size, delay_s), shift(panel.bid_size, delay_s)
    exit_bid, exit_ask = shift(panel.bid, delay_s + hold_s), shift(panel.ask, delay_s + hold_s)
    live = tradable & (panel.tau >= tau_min) & (panel.tau - delay_s >= 1)

    rows = []
    for side, entry, room, out in ((1, ask_in, ask_room, exit_bid), (-1, bid_in, bid_room, exit_ask)):
        with np.errstate(invalid="ignore"):
            take = (live & (side * jump >= theta) & (side * follow < follow_max * side * jump)
                    & np.isfinite(entry) & (room > 0))
        m, i = np.nonzero(take)
        if len(m) == 0:
            continue
        price_in = entry[m, i]
        qty = np.minimum(size, room[m, i])
        # Close at the opposite touch after hold_s seconds, or at settlement if the market has
        # ended by then or nobody is quoting that side.
        price_out = out[m, i]
        settled = ~np.isfinite(price_out) | (panel.tau[m, i] - delay_s - hold_s < 1)
        price_out = np.where(settled, panel.payoff[m], price_out)
        fee = fee_rate * (price_in * (1 - price_in) + np.where(settled, 0.0, price_out * (1 - price_out))) * qty
        rows.append(pd.DataFrame({
            "market": m, "second": i, "side": side, "qty": qty, "price_in": price_in, "price_out": price_out,
            "settled": settled, "jump": jump[m, i], "follow": follow[m, i],
            "pnl": side * (price_out - price_in) * qty - fee,
        }))
    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def max_position(panel, trades, hold_s):
    """Largest number of shares held at once in any market (trades overlap while they are open)."""
    if len(trades) == 0:
        return 0.0
    held = np.zeros((len(panel.slug), panel.tau.shape[1] + hold_s + 1))
    m, i = trades["market"].to_numpy(), trades["second"].to_numpy()
    signed = (trades["side"] * trades["qty"]).to_numpy()
    np.add.at(held, (m, i), signed)
    np.add.at(held, (m, i + hold_s), -signed)
    return float(np.abs(np.cumsum(held, axis=1)).max())


def summarize(panel, trades, n_boot=2000):
    """(statistics, PnL per market, PnL per day) for one run."""
    per_market = np.zeros(len(panel.slug))
    if len(trades):
        np.add.at(per_market, trades["market"].to_numpy(), trades["pnl"].to_numpy())
    mean, lo, hi, se = block_bootstrap(per_market, np.ones(len(per_market)), panel.start // HOUR_S, n_boot)
    by_day = pd.Series(per_market).groupby(panel.day).sum()
    by_hour = pd.Series(per_market).groupby(panel.start // HOUR_S).sum()
    shares = float(trades["qty"].sum()) if len(trades) else 0.0
    return {
        "markets": int(len(per_market)),
        "trades": int(len(trades)),
        "trades_per_market": len(trades) / max(1, len(per_market)),
        "shares": shares,
        "pnl": float(per_market.sum()),
        "pnl_per_market": mean, "ci_lo": lo, "ci_hi": hi,
        "t_stat": mean / se if se and se > 0 else float("nan"),
        "pnl_sd": float(per_market.std(ddof=1)) if len(per_market) > 1 else float("nan"),
        "cents_per_share": float(trades["pnl"].sum()) / shares * 100 if shares > 0 else float("nan"),
        "trade_win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else float("nan"),
        "trade_loss_rate": float((trades["pnl"] < 0).mean()) if len(trades) else float("nan"),
        "market_win_rate": float((per_market > 0).mean()),
        "market_loss_rate": float((per_market < 0).mean()),
        "days_profitable": int((by_day > 0).sum()), "days": int(len(by_day)),
        "hours_profitable": float((by_hour > 0).mean()),
        "worst_day": float(by_day.min()) if len(by_day) else float("nan"),
        "worst_hour": float(by_hour.min()) if len(by_hour) else float("nan"),
        "worst_market": float(per_market.min()),
        "settled_share": float(trades["settled"].mean()) if len(trades) else float("nan"),
    }, per_market, by_day
