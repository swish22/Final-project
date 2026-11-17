"""Replay a quoter through one market's recorded book and Up-mapped trades.

Timeline for the second that starts at t:
  1. The quote is computed from the grid row at t - delta (prices, book, model output).
  2. It reaches the exchange at t as a post-only order. If it would cross the book as
     it actually stands at t, it is rejected and that side has no quote this second.
     Otherwise it rests, behind the size displayed at its price at t. A quote whose
     price did not change keeps its place in the queue.
  3. Taker prints in (t, t + 1] can fill it. A taker sell hits the bid; a taker buy
     hits the ask.
  4. If the far side of the recorded book reaches our price during (t, t + 1], the
     order that did that would have matched ours first, so we are filled in full.
     These are the fills a slow quote gets when the market moves through it.
  5. The queue is updated from the displayed size at t + 1.

Inventory left at the end settles at 1 if Up won and 0 otherwise.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import be_nice, mills_array
from grid import usable_trades
from mm.queue_fill import INF_QUEUE, end_of_second, visible_size
from mm.quoter import NONE, Quoter

FEE_RATE = 0.07
REBATE_SHARE = 0.20
SUMMARY_COLS = [
    "pnl", "n_fills", "n_cross", "shares", "spread_capture", "markout_5s", "markout_30s",
    "fee_equiv", "quote_any_s", "quote_both_s", "spread_sum", "rejects", "max_abs_inv",
    "final_inv", "mean_abs_inv", "min_equity", "seconds",
]
FILL_COLS = ["t", "ts", "side", "price", "size", "kind", "mid", "inventory"]


def market_arrays(frame, delta_s):
    """Everything the replay needs that does not depend on the model, as plain lists."""
    f = frame.sort_values("t")
    n = len(f) - 1
    d = int(delta_s)

    def lagged(values, fill):
        values = np.asarray(values)
        if d <= 0:
            return values[:n].tolist()
        out = np.concatenate([np.full(d, fill, dtype=values.dtype), values[:-d]])
        return out[:n].tolist()

    px = {c: mills_array(f[c].to_numpy()) for c in ("up_bid1", "up_bid2", "up_ask1", "up_ask2")}
    sz = {c: np.nan_to_num(f[c + "_size"].to_numpy(dtype=float)) for c in px}
    mid = f["pm_mid_filled"].ffill().to_numpy(dtype=float)
    return {
        "n": n,
        "t0": int(f["t"].iloc[0]),
        "tau": f["tau"].to_numpy(dtype=float)[:n].tolist(),
        "sigma": lagged(f["sigma"].to_numpy(dtype=float), np.nan),
        "S": lagged(f["S"].to_numpy(dtype=float), np.nan),
        "seen_bid": lagged(px["up_bid1"], NONE),
        "seen_ask": lagged(px["up_ask1"], NONE),
        "b1": px["up_bid1"].tolist(),
        "a1": px["up_ask1"].tolist(),
        "bid_lv": list(zip(px["up_bid1"].tolist(), sz["up_bid1"].tolist(),
                           px["up_bid2"].tolist(), sz["up_bid2"].tolist())),
        "ask_lv": list(zip(px["up_ask1"].tolist(), sz["up_ask1"].tolist(),
                           px["up_ask2"].tolist(), sz["up_ask2"].tolist())),
        "hi_bid": mills_array(f["hi_bid_next"].to_numpy())[:n].tolist(),
        "lo_ask": mills_array(f["lo_ask_next"].to_numpy())[:n].tolist(),
        "mid": mid.tolist(),
        "payoff": 1.0 if int(f["y"].iloc[0]) == 1 else 0.0,
    }


def decision_arrays(frame, p_col, dp_col, delta_s, n):
    f = frame.sort_values("t")
    d = int(delta_s)
    out = []
    for col in (p_col, dp_col):
        values = f[col].to_numpy(dtype=float)
        if d > 0:
            values = np.concatenate([np.full(d, np.nan), values[:-d]])
        out.append(values[:n].tolist())
    return out


def bucket_trades(trades, t0, n):
    """Taker prints grouped by the second they can fill: (t, t + 1] -> index t - t0."""
    tr = usable_trades(trades)
    grouped = {}
    if len(tr) == 0:
        return grouped
    ts = tr["ts"].to_numpy(dtype=float)
    index = (np.ceil(ts - 1e-9) - 1).astype(np.int64) - t0
    sells = (tr["up_side"].to_numpy() == "sell").tolist()
    prices = mills_array(tr["up_price"].to_numpy()).tolist()
    sizes = tr["size"].to_numpy(dtype=float).tolist()
    for i, stamp, sell, price, size in zip(index.tolist(), ts.tolist(), sells, prices, sizes):
        if 0 <= i < n and price != NONE:
            grouped.setdefault(i, []).append((stamp, sell, price, size))
    return grouped


def replay_core(a, p, dp, trades, quoter, fill_rule, want_fills=False, want_path=False):
    """Returns (summary dict, fills list, per-second equity list)."""
    n, payoff = a["n"], a["payoff"]
    tau, sigma, spot = a["tau"], a["sigma"], a["S"]
    seen_bid, seen_ask = a["seen_bid"], a["seen_ask"]
    b1, a1, bid_lv, ask_lv = a["b1"], a["a1"], a["bid_lv"], a["ask_lv"]
    hi_bid, lo_ask, mid, t0 = a["hi_bid"], a["lo_ask"], a["mid"], a["t0"]
    quote = quoter.quote_mills
    generous = fill_rule == "generous"
    rule = "generous" if generous else "cautious"

    cash = inventory = 0.0
    b_px, b_sz, b_q, b_shown, b_traded = NONE, 0.0, 0.0, None, 0.0
    a_px, a_sz, a_q, a_shown, a_traded = NONE, 0.0, 0.0, None, 0.0
    fills, path = [], []
    quote_any = quote_both = rejects = 0
    spread_sum = abs_inv_sum = max_abs_inv = 0.0
    min_equity = 0.0
    last_mid = 0.5

    for i in range(n):
        qb, qa, qbs, qas = quote(p[i], sigma[i], tau[i], inventory, spot[i], dp[i],
                                 seen_bid[i], seen_ask[i])
        if qb != NONE and a1[i] != NONE and qb >= a1[i]:
            qb = NONE
            rejects += 1
        if qa != NONE and b1[i] != NONE and qa <= b1[i]:
            qa = NONE
            rejects += 1

        if qb == NONE:
            b_px, b_sz = NONE, 0.0
        elif qb == b_px and b_sz > 0:
            b_sz = min(b_sz, qbs)
        else:
            b_px, b_sz, b_traded = qb, qbs, 0.0
            b_shown = visible_size(qb, bid_lv[i], "bid")
            b_q = INF_QUEUE if b_shown is None else b_shown
        if qa == NONE:
            a_px, a_sz = NONE, 0.0
        elif qa == a_px and a_sz > 0:
            a_sz = min(a_sz, qas)
        else:
            a_px, a_sz, a_traded = qa, qas, 0.0
            a_shown = visible_size(qa, ask_lv[i], "ask")
            a_q = INF_QUEUE if a_shown is None else a_shown

        if b_px != NONE or a_px != NONE:
            quote_any += 1
            if b_px != NONE and a_px != NONE:
                quote_both += 1
                spread_sum += (a_px - b_px) / 1000.0

        here = trades.get(i)
        if here:
            for stamp, sell, price, size in here:
                if sell:
                    if b_px == NONE or b_sz <= 0 or price > b_px:
                        continue
                    if price < b_px:
                        fill = min(b_sz, size)
                        b_q = 0.0
                    else:
                        b_traded += size
                        if b_q == INF_QUEUE:
                            continue
                        if b_q >= size - 1e-12:
                            b_q = max(0.0, b_q - size)
                            continue
                        fill = min(b_sz, size - b_q)
                        b_q = 0.0
                    b_sz -= fill
                    cash -= b_px / 1000.0 * fill
                    inventory += fill
                    fills.append((i, stamp, 1, b_px, fill, "trade", inventory))
                else:
                    if a_px == NONE or a_sz <= 0 or price < a_px:
                        continue
                    if price > a_px:
                        fill = min(a_sz, size)
                        a_q = 0.0
                    else:
                        a_traded += size
                        if a_q == INF_QUEUE:
                            continue
                        if a_q >= size - 1e-12:
                            a_q = max(0.0, a_q - size)
                            continue
                        fill = min(a_sz, size - a_q)
                        a_q = 0.0
                    a_sz -= fill
                    cash += a_px / 1000.0 * fill
                    inventory -= fill
                    fills.append((i, stamp, -1, a_px, fill, "trade", inventory))

        if b_px != NONE and b_sz > 1e-9 and lo_ask[i] != NONE and lo_ask[i] <= b_px:
            cash -= b_px / 1000.0 * b_sz
            inventory += b_sz
            fills.append((i, t0 + i + 1.0, 1, b_px, b_sz, "cross", inventory))
            b_sz = 0.0
        if a_px != NONE and a_sz > 1e-9 and hi_bid[i] != NONE and hi_bid[i] >= a_px:
            cash += a_px / 1000.0 * a_sz
            inventory -= a_sz
            fills.append((i, t0 + i + 1.0, -1, a_px, a_sz, "cross", inventory))
            a_sz = 0.0

        if b_px != NONE and b_sz > 1e-9:
            shown = visible_size(b_px, bid_lv[i + 1], "bid")
            b_q = end_of_second(b_q, b_shown, shown, b_traded, rule)
            b_shown, b_traded = shown, 0.0
        else:
            b_sz = 0.0
        if a_px != NONE and a_sz > 1e-9:
            shown = visible_size(a_px, ask_lv[i + 1], "ask")
            a_q = end_of_second(a_q, a_shown, shown, a_traded, rule)
            a_shown, a_traded = shown, 0.0
        else:
            a_sz = 0.0

        m = mid[i + 1]
        if m == m:
            last_mid = m
        equity = cash + inventory * last_mid
        if equity < min_equity:
            min_equity = equity
        held = abs(inventory)
        abs_inv_sum += held
        if held > max_abs_inv:
            max_abs_inv = held
        if want_path:
            path.append(equity)

    def mark(j):
        if j >= n:
            return payoff
        value = mid[j]
        return value if value == value else payoff

    shares = capture = mo5 = mo30 = fee_equiv = 0.0
    n_cross = 0
    rows = []
    for i, stamp, side, price_m, size, kind, inv_after in fills:
        price = price_m / 1000.0
        before = mid[i]
        if before != before:
            before = price
        shares += size
        capture += side * (before - price) * size
        mo5 += side * (mark(i + 6) - price) * size
        mo30 += side * (mark(i + 31) - price) * size
        fee_equiv += size * price * (1.0 - price)
        n_cross += kind == "cross"
        if want_fills:
            rows.append((t0 + i, stamp, "buy" if side == 1 else "sell", price, size, kind, before, inv_after))

    summary = {
        "pnl": cash + inventory * payoff,
        "n_fills": len(fills),
        "n_cross": n_cross,
        "shares": shares,
        "spread_capture": capture,
        "markout_5s": mo5,
        "markout_30s": mo30,
        "fee_equiv": fee_equiv,
        "quote_any_s": quote_any,
        "quote_both_s": quote_both,
        "spread_sum": spread_sum,
        "rejects": rejects,
        "max_abs_inv": max_abs_inv,
        "final_inv": inventory,
        "mean_abs_inv": abs_inv_sum / n if n else 0.0,
        "min_equity": min_equity,
        "seconds": n,
    }
    return summary, rows, path


def replay_market(grid_m, trades_m, quoter, fill_rule, delta_s, p_col="p", dp_col="dp_dS"):
    """One market from dataframes. Returns (summary dict, fills dataframe, equity list)."""
    base = market_arrays(grid_m, delta_s)
    p, dp = decision_arrays(grid_m, p_col, dp_col, delta_s, base["n"])
    trades = bucket_trades(trades_m, base["t0"], base["n"])
    summary, rows, path = replay_core(base, p, dp, trades, quoter, fill_rule, True, True)
    return summary, pd.DataFrame(rows, columns=FILL_COLS), path


# ---------------------------------------------------------------- many markets, many settings

def make_quoter(spec):
    return Quoter(
        h_min=spec["h_min"], kappa=spec["kappa"], gamma=spec["gamma"], size=spec["size"],
        inv_cap=spec["inv_cap"], tau_stop=spec["tau_stop"], delta_s=spec["delta_s"],
    )


def _task(args):
    parquet_dir, frame, specs, want_fills, want_paths = args
    root = Path(parquet_dir) / "trades"
    summaries, fills, paths = [], [], []
    quoters = [make_quoter(spec) for spec in specs]
    for slug, f in frame.groupby("slug", sort=False):
        trades_m = pd.read_parquet(root / f"{slug}.parquet", columns=["ts", "up_side", "up_price", "size", "paired"])
        bases, decisions, buckets = {}, {}, None
        for spec, quoter in zip(specs, quoters):
            delta = spec["delta_s"]
            if delta not in bases:
                bases[delta] = market_arrays(f, delta)
            base = bases[delta]
            if buckets is None:
                buckets = bucket_trades(trades_m, base["t0"], base["n"])
            key = (spec["model"], delta)
            if key not in decisions:
                decisions[key] = decision_arrays(f, f"p_{spec['model']}", f"dp_{spec['model']}", delta, base["n"])
            p, dp = decisions[key]
            summary, rows, path = replay_core(base, p, dp, buckets, quoter, spec["fill_rule"],
                                              want_fills, want_paths)
            summaries.append((spec["label"], slug, *[summary[c] for c in SUMMARY_COLS]))
            if want_fills and rows:
                one = pd.DataFrame(rows, columns=FILL_COLS)
                one.insert(0, "slug", slug)
                one.insert(0, "label", spec["label"])
                fills.append(one)
            if want_paths:
                paths.append(pd.DataFrame({
                    "label": spec["label"], "slug": slug,
                    "t": base["t0"] + np.arange(1, base["n"] + 1), "equity": path,
                }))
    out = pd.DataFrame(summaries, columns=["label", "slug", *SUMMARY_COLS])
    return (out,
            pd.concat(fills, ignore_index=True) if fills else None,
            pd.concat(paths, ignore_index=True) if paths else None)


def run_specs(frame, specs, parquet_dir, workers=8, want_fills=False, want_paths=False, batch=8):
    """Replay every market in `frame` under every spec. Returns (summaries, fills, paths).

    A spec is a dict with label, model, fill_rule, delta_s and the quoter settings.
    `frame` needs the grid columns plus p_<model> and dp_<model> for each model used.
    """
    models = sorted({spec["model"] for spec in specs})
    cols = ["slug", "t", "tau", "y", "sigma", "S", "pm_mid_filled", "hi_bid_next", "lo_ask_next",
            "up_bid1", "up_bid1_size", "up_bid2", "up_bid2_size",
            "up_ask1", "up_ask1_size", "up_ask2", "up_ask2_size"]
    cols += [f"{kind}_{m}" for m in models for kind in ("p", "dp")]
    slim = frame[cols]
    slugs = slim["slug"].drop_duplicates().tolist()
    groups = dict(tuple(slim.groupby("slug", sort=False)))
    tasks = [
        (str(parquet_dir), pd.concat([groups[s] for s in slugs[i:i + batch]]), specs, want_fills, want_paths)
        for i in range(0, len(slugs), batch)
    ]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=be_nice) as pool:
            results = list(pool.map(_task, tasks))
    else:
        results = [_task(task) for task in tasks]
    summaries = pd.concat([r[0] for r in results], ignore_index=True)
    fills = [r[1] for r in results if r[1] is not None]
    paths = [r[2] for r in results if r[2] is not None]
    return (summaries,
            pd.concat(fills, ignore_index=True) if fills else pd.DataFrame(),
            pd.concat(paths, ignore_index=True) if paths else pd.DataFrame())
