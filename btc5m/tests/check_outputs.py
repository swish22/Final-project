"""Independent checks of a finished test run, read back from its output files.

usage:  python tests/check_outputs.py [output folder, default output/final] [config, default config.yaml]

Recomputes the headline numbers a second way and checks the replay's bookkeeping:
every fill is legal, inventory respects the cap, and each strategy's PnL equals the
settlement value of its fills.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT / (sys.argv[1] if len(sys.argv) > 1 else "output/final")).resolve()
cfg = yaml.safe_load((ROOT / (sys.argv[2] if len(sys.argv) > 2 else "config.yaml")).read_text())
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


pred = pd.read_parquet(OUT / "predictions.parquet")
scores = pd.read_csv(OUT / "forecast_scores.csv")
fills = pd.read_parquet(OUT / "fills.parquet")
pnl = pd.read_csv(OUT / "market_pnl.csv")
metrics = pd.read_csv(OUT / "mm_metrics.csv")
markets = pd.concat([pd.read_parquet(p) for p in (ROOT / cfg["paths"]["parquet_dir"] / "markets").glob("*.parquet")])
markets = markets.set_index("slug")

# ---- the test set
test_days = cfg["split"]["test"]
slugs = pnl["slug"].unique()
days = markets.loc[slugs, "day_idx"]
check("every replayed market is on a test day", bool(days.between(*test_days).all()),
      f"{len(slugs)} markets, days {days.min()}-{days.max()}")
check("forecasts are scored on test days only", set(pred["slug"]) <= set(slugs) and set(pred["split"]) == {"test"})
check("the config was frozen before the test run", bool(cfg.get("frozen")))

# ---- forecasts, recomputed from the saved predictions
official = (markets.loc[pred["slug"], "winner"].to_numpy() == "up").astype(float)
check("labels match the official winners", bool((official == pred["y"].to_numpy()).all()))
slow = max(cfg["latency"]["scenarios"])
table = scores.loc[(scores["latency_s"] == slow) & (scores["tau_bucket"] == "all")].set_index("model")
models = cfg["models"]["scored"]
check("every model is scored on the same rows", table["n"].nunique() == 1 and not pred[models].isna().any().any(),
      f"{int(table['n'].iloc[0]):,} market-seconds")
for model in models:
    again = float(np.mean((pred[model] - pred["y"]) ** 2))
    check(f"Brier of {model} recomputed", abs(again - table.loc[model, "brier"]) < 1e-9, f"{again:.4f}")
check("probabilities stay inside the clip", bool(pred[models].min().min() >= 0.005 - 1e-12
                                                  and pred[models].max().max() <= 0.995 + 1e-12))

# ---- fills
payoff = (markets.loc[fills["slug"], "winner"].to_numpy() == "up").astype(float)
start = markets.loc[fills["slug"], "start_unix"].to_numpy()
end = markets.loc[fills["slug"], "end_unix"].to_numpy()
sign = np.where(fills["side"] == "buy", 1.0, -1.0)
cents = fills["price"].to_numpy() * 100
check("fill prices are whole cents between 1 and 99",
      bool(np.allclose(cents, np.rint(cents), atol=1e-6) and cents.min() >= 1 - 1e-6 and cents.max() <= 99 + 1e-6))
check("fills happen inside the market window", bool(((fills["ts"] > start) & (fills["ts"] <= end)).all()))
check("no fill is larger than the quote size", bool(fills["size"].max() <= cfg["quote"]["size"] + 1e-9))
check("inventory never exceeds the cap", bool(fills["inventory"].abs().max() <= cfg["quote"]["inv_cap"] + 1e-6),
      f"largest {fills['inventory'].abs().max():.1f}")

fills = fills.assign(value=sign * (payoff - fills["price"].to_numpy()) * fills["size"].to_numpy(),
                     signed=sign * fills["size"].to_numpy(), left=end - fills["t"].to_numpy())
by_fill = fills.groupby("label")["value"].sum()
by_market = pnl.groupby("label")["pnl"].sum()
worst = float((by_fill.reindex(by_market.index).fillna(0.0) - by_market).abs().max())
check("each strategy's PnL equals the settlement value of its fills", worst < 1e-6, f"largest gap ${worst:.2e}")
per = fills.groupby(["label", "slug"])["value"].sum()
joined = pnl.set_index(["label", "slug"])["pnl"].to_frame().join(per).fillna(0.0)
check("and market by market", float((joined["pnl"] - joined["value"]).abs().max()) < 1e-6)
final = fills.groupby(["label", "slug"])["signed"].sum()
joined = pnl.set_index(["label", "slug"])["final_inv"].to_frame().join(final).fillna(0.0)
check("final inventory equals the sum of signed fills", float((joined["final_inv"] - joined["signed"]).abs().max()) < 1e-6)

for label, g in fills.groupby("label"):
    name, _, delta = label.split("|")
    tuned = str(delta) if str(delta) in cfg["quote"]["params"] else str(slow)
    stop = cfg["quote"]["params"][tuned][name]["tau_stop"]
    # A quote posted at the last allowed second can still fill during that second.
    if not (g["left"] >= stop).all():
        check(f"{label}: no fill after quoting stopped", False, f"earliest {g['left'].min()} s left, stop {stop}")
        break
else:
    check("no fill after a strategy stopped quoting", True)

total = metrics.set_index("label")["pnl"]
check("the metrics table matches the per-market table", float((total - by_market.reindex(total.index)).abs().max()) < 1e-6)

print()
print("all checks passed" if not failures else f"{len(failures)} check(s) FAILED: {failures}")
sys.exit(1 if failures else 0)
