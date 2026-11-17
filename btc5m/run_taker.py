"""The taker strategy: choose its settings on the learning days, then score the test days once.

Usage (from this directory, after `python run_all.py test` has been run):
    python run_taker.py tune     # choose on days 0-10, confirm on days 11-13, freeze into config.yaml
    python run_taker.py test     # score days 14-16 with the frozen settings

`tune` only ever simulates train and validation markets. `test` refuses to run until
`tune` has frozen the settings.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_all  # noqa: E402
from common import be_nice  # noqa: E402
from eval.plots import plot_taker  # noqa: E402
from mm.taker import COLUMNS, Panel, max_position, simulate, summarize  # noqa: E402
from models.d1 import D1  # noqa: E402

GRID = {
    "theta": [0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08],
    "follow_max": [0.25, 0.5, 0.75, float("inf")],
    "hold_s": [1, 2, 3, 5, 10],
    "tau_min": [10, 30],
}
KEYS = ("theta", "follow_max", "hold_s", "tau_min", "size")
MIN_TRADES_PER_MARKET = 1.0      # a rule that almost never trades cannot be called consistent
SCENARIOS = [("as assumed: no delay, no fees", {}),
             ("order arrives 1 s late", {"delay_s": 1}),
             ("order arrives 2 s late", {"delay_s": 2}),
             ("with Polymarket's taker fee", {"fee_rate": 0.07}),
             ("1 s late and with the taker fee", {"delay_s": 1, "fee_rate": 0.07})]


def load_frame(cfg):
    """Grid with the BTC model's probability, cached. Built exactly as the test stage builds it."""
    out = (ROOT / cfg["paths"]["output_dir"]).resolve()
    path = out / "cache" / "taker_frame.parquet"
    if path.exists():
        return pd.read_parquet(path), out
    ctx = run_all.prepare(cfg, ("train", "val", "test"))
    kept = ctx["kept"]
    frame, _ = run_all.build_frame(cfg, ctx, kept.loc[kept["split"].isin(("train", "val")), "slug"])
    clip = (float(cfg["model"]["clip_lo"]), float(cfg["model"]["clip_hi"]))
    frame["p_D1"] = D1(*clip).predict(frame).to_numpy()
    frame = frame[COLUMNS]
    frame.to_parquet(path, index=False, compression="zstd")
    return frame, out


def panel_for(frame, split):
    return Panel(frame.loc[frame["split"] == split])


def chosen_settings(cfg):
    taker = cfg.get("taker") or {}
    if not taker.get("frozen"):
        raise SystemExit("run `python run_taker.py tune` first: the taker settings are not frozen")
    return {key: float(taker[key]) if key != "hold_s" and key != "tau_min" else int(taker[key]) for key in KEYS}


def stage_tune(cfg, cfg_path):
    frame, out = load_frame(cfg)
    train, val = panel_for(frame, "train"), panel_for(frame, "val")
    run_all.log(f"choosing on {len(train.slug)} train markets, confirming on {len(val.slug)} validation markets")
    rows = []
    for theta, follow_max, hold_s, tau_min in product(*GRID.values()):
        stats, _, _ = summarize(train, simulate(train, theta, follow_max, hold_s, tau_min), n_boot=200)
        rows.append({"theta": theta, "follow_max": follow_max, "hold_s": hold_s, "tau_min": tau_min, **stats})
    table = pd.DataFrame(rows)
    table["sharpe"] = table["pnl_per_market"] / table["pnl_sd"]
    table = table.sort_values("sharpe", ascending=False).reset_index(drop=True)
    table.to_csv(out / "taker_tune.csv", index=False)
    best = table.loc[table["trades_per_market"] >= MIN_TRADES_PER_MARKET].iloc[0]
    chosen = {"theta": float(best["theta"]), "follow_max": float(best["follow_max"]),
              "hold_s": int(best["hold_s"]), "tau_min": int(best["tau_min"]), "size": 10.0}
    run_all.log("chosen on train: " + ", ".join(f"{a}={b:g}" for a, b in chosen.items()))

    report = []
    for name, panel in (("train", train), ("val", val)):
        stats, _, _ = summarize(panel, simulate(panel, **chosen))
        report.append({"split": name, **stats})
        run_all.log(f"  {name}: {stats['pnl_per_market']:+.3f} per market [{stats['ci_lo']:+.3f}, {stats['ci_hi']:+.3f}], "
                    f"{stats['trades_per_market']:.1f} trades per market, "
                    f"{stats['days_profitable']}/{stats['days']} days profitable")
    pd.DataFrame(report).to_csv(out / "taker_dev.csv", index=False)
    cfg["taker"] = {**chosen, "frozen": True}
    run_all._write_config(cfg_path, cfg)
    run_all.log(f"froze the taker settings in {cfg_path.name}")


def stage_test(cfg, cfg_path):
    chosen = chosen_settings(cfg)
    frame, out = load_frame(cfg)
    rows, days, curves = [], [], {}
    for split in ("train", "val", "test"):
        panel = panel_for(frame, split)
        for name, extra in SCENARIOS:
            trades = simulate(panel, **chosen, **extra)
            stats, per_market, by_day = summarize(panel, trades)
            stats["max_position"] = max_position(panel, trades, chosen["hold_s"])
            rows.append({"split": split, "scenario": name, **stats})
            if split == "test":
                curves[name] = per_market[np.argsort(panel.start)]
            if not extra:
                days.append(pd.DataFrame({"split": split, "day": by_day.index, "pnl": by_day.to_numpy()}))
                if split == "test":
                    trades.assign(slug=panel.slug[trades["market"].to_numpy()]).drop(columns="market") \
                        .to_parquet(out / "taker_trades.parquet", index=False)
                    pd.DataFrame({"slug": panel.slug, "start_unix": panel.start, "pnl": per_market}) \
                        .to_csv(out / "taker_market_pnl.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "taker_metrics.csv", index=False)
    by_day = pd.concat(days, ignore_index=True)
    by_day.to_csv(out / "taker_days.csv", index=False)
    plot_taker(curves, table.loc[table["split"] == "test"], out / "plots" / "taker.png")
    (out / "taker_results.md").write_text(render(cfg, chosen, table, by_day), encoding="utf-8")
    for row in table.loc[table["split"] == "test"].itertuples(index=False):
        print(f"  test, {row.scenario:34} {row.pnl_per_market:+.3f} per market "
              f"[{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]  {row.days_profitable}/{row.days} days profitable")
    run_all.log(f"wrote taker tables to {out}")


def render(cfg, chosen, table, by_day):
    """taker_results.md"""
    def block(rows, columns, formats):
        head = "| " + " | ".join(columns.values()) + " |"
        rule = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(formats.get(c, "{}").format(getattr(r, c)) for c in columns) + " |"
                for r in rows.itertuples(index=False)]
        return "\n".join([head, rule, *body])

    money = {"pnl": "{:+,.0f}", "pnl_per_market": "{:+.2f}", "ci_lo": "{:+.2f}", "ci_hi": "{:+.2f}",
             "t_stat": "{:.1f}", "trades_per_market": "{:.1f}", "cents_per_share": "{:+.2f}",
             "trade_win_rate": "{:.0%}", "market_win_rate": "{:.0%}", "market_loss_rate": "{:.0%}",
             "hours_profitable": "{:.0%}", "worst_day": "{:+,.0f}", "worst_market": "{:+.1f}",
             "max_position": "{:.0f}", "pnl_sd": "{:.2f}"}
    base = table.loc[table["scenario"] == SCENARIOS[0][0]].assign(
        days=lambda d: d["days_profitable"].astype(str) + " of " + d["days"].astype(str))
    test = table.loc[table["split"] == "test"].assign(
        days=lambda d: d["days_profitable"].astype(str) + " of " + d["days"].astype(str))
    follow = "any amount" if not np.isfinite(chosen["follow_max"]) else f"less than {chosen['follow_max']:.0%} of that move"
    lines = [
        "# Taker strategy results",
        "",
        "## The rule",
        "",
        f"Each second, if the BTC model's probability moved by at least {chosen['theta']:g} in the last second "
        f"and the Polymarket mid has followed by {follow}, trade {chosen['size']:g} shares in the direction of the move "
        "at the displayed price (buy at the ask or sell at the bid). Close the trade "
        f"{chosen['hold_s']} seconds later at the opposite touch. No new trades in the last {chosen['tau_min']} seconds, "
        "on a one-sided book, or in the second the strike is first recorded.",
        "",
        "## Assumptions",
        "",
        "- Zero reaction time: the decision at second t uses data up to t and fills against the book displayed at t.",
        "- No fees.",
        "- The displayed size is available to us, and our trades do not move the market.",
        "",
        "These were chosen to see whether the signal exists at all. The last table shows what each one is worth.",
        "",
        "## Results under those assumptions",
        "",
        "The settings were chosen on the train days only. Validation and test days played no part in choosing them, "
        "and the test days were scored once.",
        "",
        block(base, {"split": "days", "markets": "markets", "trades_per_market": "trades per market",
                     "pnl": "total PnL ($)", "pnl_per_market": "$ per market", "ci_lo": "low", "ci_hi": "high",
                     "t_stat": "t", "cents_per_share": "cents per share", "trade_win_rate": "trades won",
                     "days": "days profitable", "hours_profitable": "hours profitable"}, money),
        "",
        "The interval is a 95% bootstrap that resamples whole hours.",
        "",
        "Risk on the same runs:",
        "",
        block(base, {"split": "days", "pnl_sd": "sd per market ($)", "market_win_rate": "markets with a profit",
                     "market_loss_rate": "markets with a loss", "worst_market": "worst market ($)",
                     "worst_day": "worst day ($)", "max_position": "largest position (shares)"}, money),
        "",
        "PnL by day:",
        "",
        block(by_day, {"split": "days", "day": "day", "pnl": "PnL ($)"}, {"pnl": "{:+,.0f}"}),
        "",
        "## What the assumptions are worth (test days)",
        "",
        block(test, {"scenario": "scenario", "pnl": "total PnL ($)", "pnl_per_market": "$ per market",
                     "ci_lo": "low", "ci_hi": "high", "cents_per_share": "cents per share",
                     "days": "days profitable"}, money),
        "",
        "The taker fee is Polymarket's formula for crypto markets, 0.07 x price x (1 - price) per share, "
        "charged on the way in and on the way out.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("tune", "test"))
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    be_nice()
    {"tune": stage_tune, "test": stage_test}[args.stage](cfg, cfg_path)


if __name__ == "__main__":
    main()
