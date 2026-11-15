"""Calibration, one market tape, and cumulative PnL."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval.mm_metrics import cumulative_wealth


def plot_calibration(lagged, models, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, label="perfect")
    for model in models:
        frame = lagged[["y", model]].dropna()
        if frame.empty:
            continue
        try:
            frame = frame.copy()
            frame["bin"] = pd.qcut(frame[model], q=8, duplicates="drop")
        except ValueError:
            frame = frame.copy()
            frame["bin"] = 0
        grouped = frame.groupby("bin", observed=True).agg(p=(model, "mean"), y=("y", "mean"))
        ax.plot(grouped["p"], grouped["y"], marker="o", lw=1.2, label=model)
    ax.set_xlabel("predicted P(Up)")
    ax.set_ylabel("realized frequency")
    ax.set_title("Test-set reliability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_example(grid, paths, slug, path):
    market = grid.loc[grid["slug"] == slug].sort_values("t")
    if market.empty:
        return
    t0 = int(market["t"].iloc[0])
    seconds = market["t"].to_numpy() - t0
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8), sharex=True)

    ax = axes[0]
    for model, column in (("market", "market"), ("G1", "G1"), ("D1", "D1"), ("D2", "D2")):
        if column in market.columns:
            ax.plot(seconds, market[column], lw=1.1, label=model)
    ax.set_ylabel("P(Up)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=False, ncol=4)
    twin = ax.twinx()
    twin.plot(seconds, market["S"], color="0.45", lw=1.0, label="S")
    twin.set_ylabel("BTC proxy")
    ax.set_title(slug)

    quote = paths.loc[
        (paths["slug"] == slug) & (paths["model"] == "D1")
        & (paths["fill_rule"] == "cautious") & ~paths["settled"].astype(bool)
    ].sort_values("t")
    ax = axes[1]
    ax.plot(seconds, market["pm_mid"], color="0.45", lw=1.0, label="mid")
    if len(quote):
        x = quote["t"].to_numpy() - t0
        ax.plot(x, pd.to_numeric(quote["bid"], errors="coerce"), lw=1.0, label="bid")
        ax.plot(x, pd.to_numeric(quote["ask"], errors="coerce"), lw=1.0, label="ask")
    ax.set_ylabel("quotes")
    ax.legend(loc="upper left", frameon=False, ncol=3)

    ax = axes[2]
    if len(quote):
        ax.step(quote["t"].to_numpy() - t0, quote["inventory"], where="post", lw=1.2)
    ax.set_ylabel("inventory")
    ax.set_xlabel("seconds from open")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cumulative(paths, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    models = ["G1", "D1", "D2", "market"]
    for ax, rule in zip(axes, ("cautious", "generous")):
        for model in models:
            group = paths.loc[
                (paths["model"] == model) & (paths["fill_rule"] == rule) & (paths["split"] == "test")
            ]
            if group.empty:
                continue
            clock, wealth = cumulative_wealth(group)
            ax.plot(clock, wealth, lw=1.3, label=model)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(rule)
        ax.set_xlabel("seconds replayed")
        ax.legend(frameon=False)
    axes[0].set_ylabel("cumulative PnL")
    fig.suptitle("Test markets")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
