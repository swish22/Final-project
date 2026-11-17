"""Validate, fit, tune on the validation days, then score the test days once.

Usage (from this directory):
    python run_all.py tune                 # fit on train, tune on validation, freeze the config
    python run_all.py test                 # score the test days with the frozen config
    python run_all.py tune --config config_pilot.yaml

`tune` never reads the test days. `test` refuses to run until `tune` has frozen the config.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from common import assign_split, be_nice, brier_score  # noqa: E402
from convert.convert import pick_rows  # noqa: E402
from eval.forecast_scores import (  # noqa: E402
    brier_differences, calibration_error, lag_predictions, score_predictions,
)
from eval.mm_metrics import pnl_differences, summarize  # noqa: E402
from eval.lead_lag import lead_lag  # noqa: E402
from eval.plots import (  # noqa: E402
    latency_name, plot_brier, plot_brier_by_tau, plot_calibration, plot_cumulative, plot_decomposition,
    plot_example, plot_frontier, plot_latency, plot_lead_lag, plot_model_vs_market, plot_pnl,
)
from grid import (  # noqa: E402
    attach_spot, basis_table, build_pm_grid, build_spot, fit_basis, load_markets, market_spot_table,
)
from mm.replay import run_specs  # noqa: E402
from models.base import MarketMid  # noqa: E402
from models.d1 import D1  # noqa: E402
from models.d2 import D2, D3  # noqa: E402
from models.g0 import G0  # noqa: E402
from models.g1 import G1  # noqa: E402
from report import render_results  # noqa: E402
from validate.checks import validate  # noqa: E402

MODEL_CLASSES = {"G0": G0, "G1": G1, "market": MarketMid, "D1": D1, "D2": D2, "D3": D3}
SPOT_MODELS = {"D1", "D2", "D3"}
QUOTE_KEYS = ("h_min", "kappa", "gamma", "tau_stop")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- data

def prepare(cfg, splits):
    """Load, validate and grid the markets whose day falls in the given splits."""
    parquet_dir = (ROOT / cfg["paths"]["parquet_dir"]).resolve()
    out = (ROOT / cfg["paths"]["output_dir"]).resolve()
    (out / "plots").mkdir(parents=True, exist_ok=True)
    workers = int(cfg.get("workers", 8))

    markets = load_markets(parquet_dir)
    days = [d for name in splits for d in cfg["split"][name]]
    markets = markets.loc[markets["day_idx"].between(min(days), max(days))]
    per_day = (cfg.get("sample") or {}).get("per_day")
    if per_day:
        index = list(csv.DictReader(open((ROOT / cfg["paths"]["index"]).resolve())))
        first = int(index[0]["start_unix"])
        chosen = {r["slug"] for r in pick_rows(index, first, (min(days), max(days)), int(per_day))}
        markets = markets.loc[markets["slug"].isin(chosen)]
    markets = markets.reset_index(drop=True)
    log(f"{len(markets)} converted markets on days {min(days)}-{max(days)}")

    clock_ok = markets["offset_iqr_ms"] <= float(cfg["validation"]["max_offset_iqr_ms"])
    spot = build_spot(parquet_dir, markets.loc[clock_ok])
    spot_table = market_spot_table(markets, spot)
    pm_grid, pm_stats = build_pm_grid(parquet_dir, markets, workers, out / "cache")
    kept, dropped, report = validate(markets, pm_stats, spot_table, cfg)
    if kept.empty:
        raise SystemExit("no markets passed validation")
    kept["split"] = assign_split(kept["day_idx"], {k: cfg["split"][k] for k in splits})
    pm_grid = pm_grid.loc[pm_grid["slug"].isin(set(kept["slug"]))].reset_index(drop=True)
    log(f"kept {len(kept)} markets, dropped {len(dropped)}; "
        + ", ".join(f"{k}: {int((kept['split'] == k).sum())}" for k in splits))
    return {
        "parquet_dir": parquet_dir, "out": out, "workers": workers, "markets": markets,
        "kept": kept, "spot": spot, "spot_table": spot_table, "pm_grid": pm_grid, "report": report,
    }


def basis_fit(cfg, ctx, proxy, fit_slugs, enabled):
    basis = basis_table(ctx["markets"], ctx["spot_table"], proxy)
    fit = fit_basis(basis, fit_slugs) if enabled else {"beta": 1.0, "sd": 0.0, "n": 0}
    return basis, fit


def build_frame(cfg, ctx, fit_slugs):
    """Full grid with the frozen (or current) spot settings. Returns (frame, basis fit)."""
    proxy = cfg["spot"]["proxy"]
    basis, fit = basis_fit(cfg, ctx, proxy, fit_slugs, bool(cfg["basis"]["enabled"]))
    frame = attach_spot(
        ctx["pm_grid"], ctx["kept"], ctx["spot"], basis, proxy,
        float(cfg["vol"]["half_life_s"]), float(cfg["vol"]["sigma_scale"]), fit["beta"], fit["sd"],
    )
    frame["split"] = frame["slug"].map(ctx["kept"].set_index("slug")["split"]).to_numpy()
    return frame.sort_values(["slug", "t"]).reset_index(drop=True), fit


def fit_predict(cfg, frame, fit_splits):
    clip = (float(cfg["model"]["clip_lo"]), float(cfg["model"]["clip_hi"]))
    names = list(dict.fromkeys([*cfg["models"]["scored"], *cfg["models"]["quoted"]]))
    train = frame.loc[frame["split"].isin(fit_splits)]
    if train.groupby("slug")["y"].first().nunique() < 2:
        raise SystemExit("the fitting markets need both Up and Down outcomes")
    models = {}
    for name in names:
        cls = MODEL_CLASSES[name]
        model = cls(float(cfg["model"]["logistic_C"]), *clip) if name in ("G1", "D2", "D3") else cls(*clip)
        models[name] = model.fit(train)
        frame[f"p_{name}"] = model.predict(frame).to_numpy()
        if name in cfg["models"]["quoted"]:
            frame[f"dp_{name}"] = model.dp_dS(frame)
    return models


def scenarios(cfg):
    """Latencies that get their own tuned settings, fastest first."""
    return sorted(int(d) for d in cfg["latency"]["scenarios"])


def strategies(cfg):
    """Quoting strategies as (name, model, tuning objective, minimum shares per market).

    A variant reuses a quoted model's forecasts with settings chosen under another
    objective or activity floor.
    """
    q = cfg["quote"]
    base, floor = q.get("objective", "sharpe"), float(q.get("min_shares_per_market", 0))
    out = [(m, m, base, floor) for m in cfg["models"]["quoted"]]
    for name, variant in (q.get("variants") or {}).items():
        out.append((name, variant["model"], variant.get("objective", base),
                    float(variant.get("min_shares_per_market", floor))))
    return out


def score(cfg, frame, splits):
    """Score every model at every latency scenario. Returns ({latency: lagged rows}, scores)."""
    names = cfg["models"]["scored"]
    renamed = frame.loc[frame["split"].isin(splits)].rename(columns={f"p_{n}": n for n in names})
    lagged, tables = {}, []
    for delta in scenarios(cfg):
        lagged[delta] = lag_predictions(renamed, names, delta)
        table = score_predictions(lagged[delta], names)
        table.insert(0, "latency_s", delta)
        tables.append(table)
    return lagged, pd.concat(tables, ignore_index=True)


def differences(cfg, lagged, split):
    names = cfg["models"]["scored"]
    tables = []
    for delta, rows in lagged.items():
        table = brier_differences(rows, names, ["market", "G1"], split)
        table.insert(0, "latency_s", delta)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------- tuning

def tune_d1(cfg, ctx):
    """Pick the proxy, the basis correction, the volatility half-life and scale by validation Brier."""
    kept, spot = ctx["kept"], ctx["spot"]
    delta = scenarios(cfg)[-1]
    train_slugs = kept.loc[kept["split"] == "train", "slug"]
    val_pm = ctx["pm_grid"].loc[ctx["pm_grid"]["slug"].isin(set(kept.loc[kept["split"] == "val", "slug"]))]
    digital = D1(float(cfg["model"]["clip_lo"]), float(cfg["model"]["clip_hi"]))
    rows, basis_rows = [], []
    for proxy in cfg["spot"]["proxy_grid"]:
        for mode in (False, True):
            basis, fit = basis_fit(cfg, ctx, proxy, train_slugs, mode)
            if mode:
                basis_rows.append({"proxy": proxy, **fit})
            g = attach_spot(val_pm, kept, spot, basis, proxy, cfg["vol"]["half_life_grid"][0], 1.0,
                            fit["beta"], fit["sd"]).sort_values(["slug", "t"]).reset_index(drop=True)
            t = g["t"].to_numpy()
            open_rows = (g["tau"] >= 1).to_numpy()
            y = g["y"].to_numpy(dtype=float)
            for half in cfg["vol"]["half_life_grid"]:
                sigma = spot.at(spot.sigma(proxy, half), t)
                for scale in cfg["vol"]["sigma_scale_grid"]:
                    g["sigma"] = sigma * float(scale)
                    p = digital.predict(g).groupby(g["slug"], sort=False).shift(delta).to_numpy()
                    ok = open_rows & np.isfinite(p)
                    rows.append({
                        "proxy": proxy, "basis": mode, "half_life_s": float(half),
                        "sigma_scale": float(scale), "val_brier": brier_score(y[ok], p[ok]), "n": int(ok.sum()),
                    })
    table = pd.DataFrame(rows).sort_values("val_brier").reset_index(drop=True)
    return table.iloc[0].to_dict(), table, pd.DataFrame(basis_rows)


def quote_specs(cfg, model, delta):
    q = cfg["quote"]
    kappas = q["kappa_grid"] if model in SPOT_MODELS else [0.0]
    specs = []
    for h_min, kappa, gamma, tau_stop in product(q["h_min_grid"], kappas, q["gamma_grid"], q["tau_stop_grid"]):
        specs.append({
            "label": f"{model}|{h_min}|{kappa}|{gamma}|{tau_stop}", "model": model, "fill_rule": "cautious",
            "delta_s": delta, "h_min": float(h_min), "kappa": float(kappa), "gamma": float(gamma),
            "tau_stop": float(tau_stop), "size": q["size"], "inv_cap": q["inv_cap"],
        })
    return specs


def tune_quotes(cfg, ctx, frame, delta):
    """Replay every quote setting of every quoted model on the validation days, cautious fills."""
    val = frame.loc[frame["split"] == "val"]
    specs = [s for model in cfg["models"]["quoted"] for s in quote_specs(cfg, model, delta)]
    log(f"latency {delta}s: replaying {len(specs)} quote settings on {val['slug'].nunique()} validation markets")
    summaries, _, _ = run_specs(val, specs, ctx["parquet_dir"], ctx["workers"])
    by = summaries.groupby("label", sort=False).agg(
        pnl=("pnl", "sum"), mean=("pnl", "mean"), sd=("pnl", "std"), shares=("shares", "sum"))
    lookup = {s["label"]: s for s in specs}
    rows = []
    for label, r in by.iterrows():
        s = lookup[label]
        rows.append({
            "latency_s": delta, "model": s["model"], **{k: s[k] for k in QUOTE_KEYS}, "val_pnl": r["pnl"],
            "val_pnl_per_market": r["mean"], "val_pnl_sd": r["sd"], "val_shares": r["shares"],
            "val_sharpe": r["mean"] / r["sd"] if r["sd"] > 0 else -np.inf,
        })
    table = pd.DataFrame(rows)
    table["val_shares_per_market"] = table["val_shares"] / val["slug"].nunique()
    return table.sort_values(["model", "val_sharpe"], ascending=[True, False]).reset_index(drop=True)


def select_quotes(cfg, table):
    """Pick each strategy's settings at each latency from the validation table.

    The objective is the Sharpe ratio of per-market PnL (mean / standard deviation), which does
    not reward settings that win by holding a large one-way position through settlement. Only
    settings that traded at least the strategy's minimum shares per market are eligible, because
    without a floor the best "market maker" is the one that never quotes.
    Returns ({latency: {strategy: settings}}, the chosen rows).
    """
    columns = {"sharpe": "val_sharpe", "pnl": "val_pnl"}
    params, chosen = {}, []
    for delta in scenarios(cfg):
        params[str(delta)] = {}
        at = table.loc[table["latency_s"] == delta]
        for name, model, objective, floor in strategies(cfg):
            g = at.loc[at["model"] == model]
            active = g.loc[g["val_shares_per_market"] >= floor]
            top = (active if len(active) else g).sort_values(columns[objective], ascending=False).iloc[0]
            params[str(delta)][name] = {k: float(top[k]) for k in QUOTE_KEYS}
            chosen.append({"strategy": name, **top.to_dict()})
            log(f"  latency {delta}s {name}: " + ", ".join(f"{k}={top[k]:g}" for k in QUOTE_KEYS)
                + f" (validation {top['val_pnl_per_market']:+.2f}/market, "
                  f"{top['val_shares_per_market']:.0f} shares/market)")
    return params, pd.DataFrame(chosen)


def freeze(cfg, cfg_path, ctx, frame, table):
    """Choose the quote settings, record how they did on validation, and freeze the config."""
    out = ctx["out"]
    cfg["quote"]["params"], chosen = select_quotes(cfg, table)
    chosen.to_csv(out / "tune_chosen.csv", index=False)
    plot_frontier(table, chosen, out / "plots" / "frontier_validation.png")
    specs = final_specs(cfg, ("cautious",), with_robustness=False)
    _, _, metrics = replay_metrics(cfg, ctx, frame.loc[frame["split"] == "val"], specs)
    metrics.to_csv(out / "dev_mm_metrics.csv", index=False)
    cfg["frozen"] = True
    _write_config(cfg_path, cfg)
    log(f"froze {cfg_path.name}; validation tables are in {out}")


def final_specs(cfg, rules, with_robustness=True):
    """Frozen settings at each tuned latency, plus slower latencies with the slowest tuned settings."""
    q = cfg["quote"]
    tuned = scenarios(cfg)
    extra = [int(d) for d in cfg["latency"].get("robustness", [])] if with_robustness else []
    specs = []
    for name, model, _, _ in strategies(cfg):
        for delta in [*tuned, *extra]:
            params = q["params"][str(delta if delta in tuned else tuned[-1])][name]
            for rule in rules:
                if delta in extra and rule != "cautious":
                    continue
                specs.append({"label": f"{name}|{rule}|{delta}", "model": model, "fill_rule": rule,
                              "delta_s": delta, **params, "size": q["size"], "inv_cap": q["inv_cap"]})
    return specs


def replay_metrics(cfg, ctx, frame, specs, want_fills=False):
    summaries, fills, _ = run_specs(frame, specs, ctx["parquet_dir"], ctx["workers"], want_fills=want_fills)
    summaries = summaries.merge(ctx["kept"][["slug", "start_unix", "split"]], on="slug", how="left")
    metrics = summarize(summaries, float(cfg["risk"]["cvar_alpha"]))
    parts = metrics["label"].str.split("|", expand=True)
    metrics.insert(1, "model", parts[0])
    metrics.insert(2, "fill_rule", parts[1])
    metrics.insert(3, "latency_s", parts[2].astype(int))
    metrics.insert(4, "tuned", metrics["latency_s"].isin(scenarios(cfg)))
    return summaries, fills, metrics


# ---------------------------------------------------------------- stages

def stage_tune(cfg, cfg_path):
    if cfg.get("frozen"):
        raise SystemExit("config is frozen; set frozen: false to tune again")
    ctx = prepare(cfg, ("train", "val"))
    out = ctx["out"]
    (out / "validation_report.md").write_text(ctx["report"], encoding="utf-8")

    log("tuning the digital model on validation Brier")
    best, d1_table, basis_fits = tune_d1(cfg, ctx)
    d1_table.to_csv(out / "tune_d1.csv", index=False)
    basis_fits.to_csv(out / "basis_fits.csv", index=False)
    cfg["spot"]["proxy"] = str(best["proxy"])
    cfg["basis"]["enabled"] = bool(best["basis"])
    cfg["vol"]["half_life_s"] = float(best["half_life_s"])
    cfg["vol"]["sigma_scale"] = float(best["sigma_scale"])
    log(f"proxy {best['proxy']}, basis correction {bool(best['basis'])}, half-life "
        f"{best['half_life_s']:g}s, sigma scale {best['sigma_scale']:g}, Brier {best['val_brier']:.5f}")

    frame, _ = build_frame(cfg, ctx, ctx["kept"].loc[ctx["kept"]["split"] == "train", "slug"])
    fit_predict(cfg, frame, ("train",))
    lagged, scores = score(cfg, frame, ("train", "val"))
    scores.to_csv(out / "dev_forecast_scores.csv", index=False)
    differences(cfg, lagged, "val").to_csv(out / "dev_brier_differences.csv", index=False)

    table = pd.concat([tune_quotes(cfg, ctx, frame, delta) for delta in scenarios(cfg)], ignore_index=True)
    table.to_csv(out / "tune_quotes.csv", index=False)
    freeze(cfg, cfg_path, ctx, frame, table)


def stage_reselect(cfg, cfg_path):
    """Choose again from the saved validation table, without replaying every setting.

    For changing the selection rule (objective or activity floor). The model settings found
    by `tune` are kept, and the test days are not read.
    """
    ctx = prepare(cfg, ("train", "val"))
    table = pd.read_csv(ctx["out"] / "tune_quotes.csv")
    frame, _ = build_frame(cfg, ctx, ctx["kept"].loc[ctx["kept"]["split"] == "train", "slug"])
    fit_predict(cfg, frame, ("train",))
    freeze(cfg, cfg_path, ctx, frame, table)


def stage_test(cfg, cfg_path):
    if not cfg.get("frozen"):
        raise SystemExit("run `tune` first: the config is not frozen")
    ctx = prepare(cfg, ("train", "val", "test"))
    out, kept = ctx["out"], ctx["kept"]
    if not (kept["split"] == "test").any():
        raise SystemExit("no test markets are converted yet")
    (out / "validation_report.md").write_text(ctx["report"], encoding="utf-8")

    learn = ("train", "val") if cfg.get("refit_on_learning_days", True) else ("train",)
    frame, basis = build_frame(cfg, ctx, kept.loc[kept["split"].isin(learn), "slug"])
    models = fit_predict(cfg, frame, learn)
    log(f"fitted on {', '.join(learn)}; scoring the test days")

    names = cfg["models"]["scored"]
    tuned = scenarios(cfg)
    lagged, scores = score(cfg, frame, ("test",))
    scores.to_csv(out / "forecast_scores.csv", index=False)
    diffs = differences(cfg, lagged, "test")
    diffs.to_csv(out / "brier_differences.csv", index=False)
    ece = pd.concat([calibration_error(rows, names, "test").assign(latency_s=d) for d, rows in lagged.items()])
    lagged[tuned[-1]].to_parquet(out / "predictions.parquet", index=False)

    quoted = cfg["models"]["quoted"]
    specs = final_specs(cfg, ("cautious", "generous"))
    everyone = [s[0] for s in strategies(cfg)]
    test = frame.loc[frame["split"] == "test"]
    log(f"replaying {len(specs)} settings on {test['slug'].nunique()} test markets")
    summaries, fills, metrics = replay_metrics(cfg, ctx, test, specs, want_fills=True)
    metrics.to_csv(out / "mm_metrics.csv", index=False)
    summaries.to_csv(out / "market_pnl.csv", index=False)
    fills.to_parquet(out / "fills.parquet", index=False)
    pairs = [(f"{a}|cautious|{d}", f"{b}|cautious|{d}") for d in tuned for a in everyone
             for b in ("market", "G1") if a != b and (a, b) != ("market", "G1")]
    pnl_diffs = pnl_differences(summaries, pairs)
    pnl_diffs.to_csv(out / "pnl_differences.csv", index=False)

    star = "D1" if "D1" in quoted else quoted[0]
    lead = lead_lag(test, star)
    lead.to_csv(out / "lead_lag.csv", index=False)
    plot_lead_lag(lead, star, out / "plots" / "lead_lag.png")
    slow = lagged[tuned[-1]]
    plots = out / "plots"
    slow_scores = scores.loc[scores["latency_s"] == tuned[-1]]
    plot_calibration(slow, [n for n in ("market", "G1", "D1") if n in names], "test", plots / "calibration.png")
    plot_brier(slow_scores, names, plots / "brier.png")
    plot_brier_by_tau(slow_scores, names, "test", plots / "brier_by_time_left.png")
    plot_pnl(metrics, everyone, plots / "pnl_per_market.png")
    plot_decomposition(metrics, everyone, plots / "fill_value.png")
    plot_latency(metrics, everyone, plots / "latency.png")
    plot_cumulative(summaries, [(f"Quoting on {latency_name(d)}", [(f"{m}|cautious|{d}", m) for m in everyone])
                                for d in tuned], plots / "cumulative_pnl.png")
    label = f"{star}|cautious|{tuned[0]}"
    example = _example_market(summaries, label)
    one = test.loc[test["slug"] == example]
    _, ex_fills, ex_path = run_specs(one, [s for s in specs if s["label"] == label], ctx["parquet_dir"], 1,
                                     want_fills=True, want_paths=True)
    plot_model_vs_market(one, example, star, plots / "model_vs_market.png")
    plot_example(one, ex_fills, ex_path, example, star, plots / "example_market.png")

    text = render_results(cfg, kept, models, basis, scores, diffs, ece, metrics, pnl_diffs, lead, example)
    (out / "results.md").write_text(text, encoding="utf-8")
    _print_headline(scores, metrics)
    log(f"wrote {out}")


def _example_market(summaries, label):
    """The market with the median PnL among those with a typical number of fills."""
    rows = summaries.loc[summaries["label"] == label]
    busy = rows.loc[rows["n_fills"] >= rows["n_fills"].median()]
    ordered = busy.sort_values("pnl")
    return ordered["slug"].iloc[len(ordered) // 2]


def _print_headline(scores, metrics):
    print("test Brier by latency:")
    wide = scores.loc[scores["tau_bucket"] == "all"].pivot(index="model", columns="latency_s", values="brier")
    print(wide.round(4).to_string())
    print("test PnL per market [95% interval]:")
    for row in metrics.sort_values(["latency_s", "fill_rule", "model"]).itertuples(index=False):
        print(f"  {row.latency_s}s {row.fill_rule:9} {row.model:7} {row.pnl_per_market:8.3f} "
              f"[{row.ci_lo:7.3f}, {row.ci_hi:7.3f}]")


def _write_config(path, cfg):
    text = yaml.safe_dump(_plain(cfg), sort_keys=False)
    path.write_text("# Tuned on the validation days, then frozen before the test days were scored.\n" + text)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return int(value) if value.is_integer() else value
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("tune", "reselect", "test"))
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    be_nice()
    {"tune": stage_tune, "reselect": stage_reselect, "test": stage_test}[args.stage](cfg, cfg_path)


if __name__ == "__main__":
    main()
