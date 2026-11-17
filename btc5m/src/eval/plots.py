"""Figures for the report and the slides."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.forecast_scores import calibration_bins
from eval.mm_metrics import wealth_curve

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 160,
})

BUCKET_ORDER = ["120+", "60-120", "30-60", "0-30"]
COLORS = {"G0": "#b8b8b8", "market": "#555555", "G1": "#e08a2e", "D1": "#1f6fb5", "D2": "#6fb0e0",
          "D3": "#8a63b8", "D1sel": "#2a9d8f"}
NAMES = {
    "G0": "Base rate (G0)", "market": "Polymarket mid", "G1": "Generic model (G1)",
    "D1": "BTC model (D1)", "D2": "BTC model, recalibrated (D2)", "D3": "BTC + market (D3)",
}
SHORT = {
    "G0": "Base rate\n(G0)", "market": "Polymarket\nmid", "G1": "Generic\nmodel (G1)",
    "D1": "BTC formula\n(D1)", "D2": "BTC recal.\n(D2)", "D3": "BTC+market\n(D3)",
}
# Quoting strategies: what each one centres its quotes on.
QUOTERS = {
    "market": "Polymarket mid", "G1": "Generic model", "D1": "BTC model, active", "D1sel": "BTC model, selective",
}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def latency_name(delta):
    return "current-second data" if int(delta) == 0 else f"data {int(delta)} s old"


# ---------------------------------------------------------------- forecasts

def plot_calibration(lagged, models, split, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls="--", label="perfect")
    for model in models:
        table = calibration_bins(lagged, model, split)
        if len(table):
            ax.plot(table["p"], table["y"], marker="o", ms=4, lw=1.4, color=COLORS[model], label=NAMES[model])
    ax.set_xlabel("predicted P(Up)")
    ax.set_ylabel("share of those seconds where Up won")
    ax.set_title("Calibration on the test days")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def plot_brier(scores, models, path, title="Forecast error on the test days"):
    """Overall Brier score per model, one bar each."""
    table = scores.loc[scores["tau_bucket"] == "all"].set_index("model")["brier"].reindex(models)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar([SHORT[m] for m in models], table.to_numpy(), color=[COLORS[m] for m in models])
    for bar, value in zip(bars, table.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_ylim(0, float(table.max()) * 1.15)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=10)
    _save(fig, path)


def plot_brier_by_tau(scores, models, split, path):
    table = scores.loc[(scores["split"] == split) & scores["tau_bucket"].isin(BUCKET_ORDER)]
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(BUCKET_ORDER))
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        values = table.loc[table["model"] == model].set_index("tau_bucket")["brier"].reindex(BUCKET_ORDER)
        ax.bar(x + (k - (len(models) - 1) / 2) * width, values, width, color=COLORS[model], label=NAMES[model])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} s left" for b in BUCKET_ORDER])
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Forecast error by time left, test days")
    ax.set_ylim(top=float(table["brier"].max()) * 1.35)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=10)
    _save(fig, path)


def plot_lead_lag(table, model, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS["D1"] if k > 0 else "0.7" if k < 0 else "0.35" for k in table["lag_s"]]
    ax.bar(table["lag_s"], table["corr"], color=colors)
    ax.set_xlabel("lead of the BTC model over the Polymarket price (seconds)")
    ax.set_ylabel("correlation of 1-second changes")
    ax.set_title("BTC moves first (blue: BTC leads, light grey: Polymarket leads)", fontsize=12)
    ax.set_xticks(table["lag_s"])
    _save(fig, path)


# ---------------------------------------------------------------- market making

def plot_pnl(metrics, models, path):
    """PnL per market with its 95% interval, for each tuned latency, cautious fills."""
    rows = metrics.loc[metrics["tuned"] & (metrics["fill_rule"] == "cautious")]
    latencies = sorted(rows["latency_s"].unique())
    fig, axes = plt.subplots(1, len(latencies), figsize=(5.4 * len(latencies), 4.2), sharey=True, squeeze=False)
    for ax, delta in zip(axes[0], latencies):
        r = rows.loc[rows["latency_s"] == delta].set_index("model").reindex(models)
        x = np.arange(len(models))
        err = np.vstack([r["pnl_per_market"] - r["ci_lo"], r["ci_hi"] - r["pnl_per_market"]])
        ax.bar(x, r["pnl_per_market"], color=[COLORS[m] for m in models], yerr=err, capsize=5,
               error_kw={"lw": 1.3})
        for xi, value, lo in zip(x, r["pnl_per_market"], r["ci_lo"]):
            ax.text(xi, min(value, lo, 0) - 0.02 * abs(rows["ci_lo"].min()) - 0.05,
                    "0.00" if abs(value) < 0.005 else f"{value:+.2f}", ha="center", va="top", fontsize=11)
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([QUOTERS[m].replace(", ", ",\n") for m in models], fontsize=10)
        ax.set_title(f"Quoting on {latency_name(delta)}")
    axes[0][0].set_ylabel("PnL per market ($), 95% interval")
    lo, hi = axes[0][0].get_ylim()
    axes[0][0].set_ylim(lo - 0.12 * (hi - lo), hi)
    _save(fig, path)


def plot_frontier(table, chosen, path):
    """Every quote setting tried on the validation days: how much it traded against what it made."""
    latencies = sorted(table["latency_s"].unique())
    fig, axes = plt.subplots(1, len(latencies), figsize=(5.8 * len(latencies), 4.5), sharey=True, squeeze=False)
    for ax, delta in zip(axes[0], latencies):
        at = table.loc[table["latency_s"] == delta]
        for model in ("market", "G1", "D1"):
            g = at.loc[at["model"] == model]
            if len(g):
                ax.scatter(g["val_shares_per_market"], g["val_pnl_per_market"], s=14, alpha=0.45,
                           color=COLORS[model], label=NAMES[model], linewidths=0)
        picks = chosen.loc[chosen["latency_s"] == delta]
        for row in picks.itertuples(index=False):
            ax.scatter(row.val_shares_per_market, row.val_pnl_per_market, s=150, marker="*",
                       color=COLORS[row.strategy], edgecolors="black", linewidths=0.9, zorder=5)
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("shares traded per market (log scale)")
        ax.set_title(f"Quoting on {latency_name(delta)}")
    axes[0][0].set_ylabel("PnL per market ($), validation days")
    axes[0][0].legend(frameon=False, fontsize=10, loc="lower left", title="one dot per setting; stars are chosen")
    _save(fig, path)


def plot_decomposition(metrics, models, path):
    """What a share was worth relative to our fill price: at the fill, 5 s later, 30 s later, at settlement."""
    rows = metrics.loc[metrics["tuned"] & (metrics["fill_rule"] == "cautious")]
    latencies = sorted(rows["latency_s"].unique())
    stages = [("spread_capture_c", "at the fill\n(spread capture)"), ("markout_5s_c", "5 s later"),
              ("markout_30s_c", "30 s later"), ("pnl_per_share_c", "at settlement")]
    fig, axes = plt.subplots(1, len(latencies), figsize=(5.6 * len(latencies), 4.3), sharey=True, squeeze=False)
    for ax, delta in zip(axes[0], latencies):
        r = rows.loc[rows["latency_s"] == delta].set_index("model")
        for model in models:
            if model in r.index:
                ax.plot(range(len(stages)), [r.loc[model, col] for col, _ in stages], marker="o", lw=1.8,
                        color=COLORS[model], label=QUOTERS[model])
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([name for _, name in stages], fontsize=10)
        ax.set_title(f"Quoting on {latency_name(delta)}")
    axes[0][0].set_ylabel("value of a filled share vs our price (cents)")
    axes[0][0].legend(frameon=False, fontsize=10)
    _save(fig, path)


def plot_latency(metrics, models, path):
    """PnL per market against reaction time, cautious fills."""
    rows = metrics.loc[metrics["fill_rule"] == "cautious"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for k, model in enumerate(models):
        r = rows.loc[rows["model"] == model].sort_values("latency_s")
        x = np.arange(len(r))
        # A small sideways shift keeps overlapping intervals readable.
        ax.errorbar(x + (k - (len(models) - 1) / 2) * 0.06, r["pnl_per_market"], yerr=[r["pnl_per_market"] - r["ci_lo"], r["ci_hi"] - r["pnl_per_market"]],
                    marker="o", lw=1.6, capsize=4, color=COLORS[model], label=QUOTERS[model])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(d)} s" for d in r["latency_s"]])
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xlabel("age of the data a quote is built from")
    ax.set_ylabel("PnL per market ($), 95% interval")
    ax.set_title("PnL against reaction time, test days")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center")
    _save(fig, path)


def plot_cumulative(summaries, panels, path):
    """One panel per (title, [(replay label, model), ...])."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.3), sharey=True, squeeze=False)
    for ax, (title, labels) in zip(axes[0], panels):
        for label, model in labels:
            rows = summaries.loc[summaries["label"] == label]
            if rows.empty:
                continue
            _, settled, _ = wealth_curve(rows)
            ax.plot(np.arange(1, len(settled) + 1), settled, lw=1.5, color=COLORS[model], label=QUOTERS[model])
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_xlabel("test markets settled, in time order")
        ax.set_title(title)
    axes[0][0].set_ylabel("cumulative PnL ($), cautious fills")
    axes[0][0].legend(frameon=False, fontsize=10)
    _save(fig, path)


def plot_taker(curves, test_rows, path):
    """Taker strategy on the test days: cumulative PnL under each assumption, and PnL per market."""
    palette = ["#1f6fb5", "#e08a2e", "#c0392b", "#8a63b8", "#555555"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for color, (name, per_market) in zip(palette, curves.items()):
        ax.plot(np.arange(1, len(per_market) + 1), np.cumsum(per_market), lw=1.6, color=color, label=name)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xlabel("test markets settled, in time order")
    ax.set_ylabel("cumulative PnL ($)")
    ax.set_title("Taker strategy on the test days")
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1]
    rows = test_rows.reset_index(drop=True)
    y = np.arange(len(rows))[::-1]
    err = np.vstack([rows["pnl_per_market"] - rows["ci_lo"], rows["ci_hi"] - rows["pnl_per_market"]])
    ax.barh(y, rows["pnl_per_market"], xerr=err, color=palette[:len(rows)], capsize=4, error_kw={"lw": 1.2})
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("as assumed: ", "").replace(" and with", "\nand") for s in rows["scenario"]], fontsize=10)
    ax.set_xlabel("PnL per market ($), 95% interval")
    ax.set_title("What each assumption is worth")
    _save(fig, path)


# ---------------------------------------------------------------- one market

def plot_model_vs_market(view, slug, model, path):
    """The idea in one picture: BTC against the strike, and the two probabilities it implies."""
    market = view.loc[view["slug"] == slug].sort_values("t")
    if market.empty:
        return
    seconds = market["t"].to_numpy() - int(market["t"].iloc[0])
    # The strike and its basis are only known a few seconds in; draw the whole path against the final ones.
    strike, basis = float(market["K"].iloc[-1]), float(market["b"].iloc[-1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot(seconds, market["S"] + basis - strike, color="tab:green", lw=1.4)
    ax.axhline(0, color="0.3", lw=0.9, ls="--")
    ax.set_ylabel("BTC minus strike ($)")
    winner = "Up" if int(market["y"].iloc[0]) == 1 else "Down"
    ax.set_title(f"One 5-minute market (resolved {winner})\nUp wins if BTC ends above the dashed line",
                 fontsize=12)
    ax = axes[1]
    ax.plot(seconds, market["pm_mid_filled"], color=COLORS["market"], lw=1.5, label="Polymarket mid")
    ax.plot(seconds, market[f"p_{model}"], color=COLORS[model], lw=1.5, label=f"{NAMES[model]}: no market data used")
    ax.set_ylabel("P(Up)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("seconds from open")
    ax.legend(frameon=False, loc="best")
    _save(fig, path)


def plot_example(view, fills, equity, slug, model, path):
    """One market: probabilities, then our fills on the mid, then inventory and equity."""
    market = view.loc[view["slug"] == slug].sort_values("t")
    if market.empty:
        return
    t0 = int(market["t"].iloc[0])
    seconds = market["t"].to_numpy() - t0
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.8), sharex=True)

    ax = axes[0]
    ax.plot(seconds, market["pm_mid_filled"], color=COLORS["market"], lw=1.2, label="Polymarket mid")
    ax.plot(seconds, market[f"p_{model}"], color=COLORS[model], lw=1.3, label=NAMES[model])
    ax.set_ylabel("P(Up)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", frameon=False, ncol=2)
    winner = "Up" if int(market["y"].iloc[0]) == 1 else "Down"
    ax.set_title(f"{slug} (resolved {winner})")

    ax = axes[1]
    ax.plot(seconds, market["pm_mid_filled"], color=COLORS["market"], lw=1.0, label="Polymarket mid")
    if len(fills):
        buys, sells = fills.loc[fills["side"] == "buy"], fills.loc[fills["side"] == "sell"]
        ax.scatter(buys["ts"] - t0, buys["price"], marker="^", s=30, color="tab:blue", label="we bought", zorder=3)
        ax.scatter(sells["ts"] - t0, sells["price"], marker="v", s=30, color="tab:red", label="we sold", zorder=3)
    ax.set_ylabel("price")
    ax.legend(loc="best", frameon=False, ncol=3)

    ax = axes[2]
    if len(fills):
        ax.step(np.r_[0, fills["ts"].to_numpy() - t0, seconds[-1]],
                np.r_[0, fills["inventory"].to_numpy(), fills["inventory"].iloc[-1]],
                where="post", lw=1.4, color="tab:purple")
    ax.set_ylabel("inventory (shares)", color="tab:purple")
    ax.set_xlabel("seconds from open")
    if len(equity):
        twin = ax.twinx()
        twin.spines["right"].set_visible(True)
        twin.plot(equity["t"].to_numpy() - t0, equity["equity"], color="tab:orange", lw=1.1)
        twin.set_ylabel("mark-to-mid equity ($)", color="tab:orange")
    _save(fig, path)
