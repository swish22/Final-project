"""Validate, fit, tune on the validation day, then score the test day once.

Usage (from this directory):
    python3 run_all.py
"""
from __future__ import annotations

import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from common import assign_split, brier_score  # noqa: E402
from eval.forecast_scores import lag_predictions, score_predictions  # noqa: E402
from eval.mm_metrics import attach_markout, summarize  # noqa: E402
from eval.plots import plot_calibration, plot_cumulative, plot_example  # noqa: E402
from grid import attach_sigma, build_grid, load_tables  # noqa: E402
from mm.quoter import Quoter  # noqa: E402
from mm.replay import replay  # noqa: E402
from models.base import MarketMid  # noqa: E402
from models.d1 import D1  # noqa: E402
from models.d2 import D2  # noqa: E402
from models.g0 import G0  # noqa: E402
from models.g1 import G1  # noqa: E402
from validate.checks import validate  # noqa: E402

QUOTED = ("G1", "D1", "D2", "market")
SCORED = ("G0", "G1", "D1", "D2", "market")


def main():
    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    parquet_dir = (ROOT / cfg["paths"]["parquet_dir"]).resolve()
    out = (ROOT / cfg["paths"]["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    print(f"loading {parquet_dir}")
    markets, book, trades, btc = load_tables(parquet_dir)
    kept, dropped, report = validate(markets, book, btc, cfg)
    (out / "validation_report.md").write_text(report)
    print(report.splitlines()[2] if len(report.splitlines()) > 2 else report)
    if kept.empty:
        raise SystemExit("no markets passed validation")

    kept["split"] = assign_split(kept["day"], cfg["split"])
    outside = kept.loc[kept["split"].isna(), "slug"].tolist()
    if outside:
        print(f"ignoring {len(outside)} markets outside the configured split")
    kept = kept.loc[kept["split"].notna()].copy()

    print("building the one-second grid")
    grid, spot = build_grid(kept, book, trades, btc, float(cfg["vol"]["half_life_s"]))
    split_of = kept.set_index("slug")["split"]
    grid["split"] = grid["slug"].map(split_of)

    book_by = {slug: frame for slug, frame in book.groupby("slug", sort=False)}
    trades_by = {slug: frame for slug, frame in trades.groupby("slug", sort=False)}
    delta = int(cfg["latency"]["delta_s"])
    clip = (float(cfg["model"]["clip_lo"]), float(cfg["model"]["clip_hi"]))

    if not cfg.get("frozen"):
        print("tuning the EWMA half-life on the validation markets")
        best_half, half_table = tune_half_life(
            grid, spot, delta, cfg["vol"]["half_life_grid"],
        )
        half_table.to_csv(out / "tune_half_life.csv", index=False)
        cfg["vol"]["half_life_s"] = best_half
        print(f"half-life {best_half:g}s")
        grid = attach_sigma(grid, spot, best_half)
        grid["split"] = grid["slug"].map(split_of)
    else:
        grid = attach_sigma(grid, spot, float(cfg["vol"]["half_life_s"]))
        grid["split"] = grid["slug"].map(split_of)
        print(f"using frozen half-life {cfg['vol']['half_life_s']}")

    grid = grid.sort_values(["slug", "t"]).reset_index(drop=True)
    train = grid.loc[grid["split"] == "train"].copy()
    if train.empty or train.groupby("slug")["y"].first().nunique() < 2:
        raise SystemExit("training split needs both Up and Down markets")

    models = fit_models(train, float(cfg["model"]["logistic_C"]), clip)
    frame = with_predictions(grid, models)

    if not cfg.get("frozen"):
        print("tuning quote parameters on validation PnL (D1, cautious fills)")
        best_quote, quote_table = tune_quotes(
            frame.loc[frame["split"] == "val"].copy(),
            book_by, trades_by, delta, cfg["quote"],
        )
        quote_table.to_csv(out / "tune_quotes.csv", index=False)
        cfg["quote"].update(best_quote)
        cfg["frozen"] = True
        _write_config(cfg_path, cfg)
        print(
            "froze "
            + ", ".join(f"{key}={best_quote[key]}" for key in ("gamma", "kappa", "h_min", "tau_stop"))
        )
    else:
        print("using frozen quote parameters")

    # Test labels are used only from here on.
    lagged = lag_predictions(frame, list(SCORED), delta)
    scores = score_predictions(lagged, SCORED)
    scores.to_csv(out / "forecast_scores.csv", index=False)

    print("replaying G1, D1, D2, and the Polymarket mid")
    fills, paths = run_replays(frame, book_by, trades_by, delta, cfg["quote"])
    fills = attach_markout(fills, grid)
    metrics, market_pnl = summarize(fills, paths, alpha=float(cfg["risk"]["cvar_alpha"]))
    metrics.to_csv(out / "mm_metrics.csv", index=False)
    market_pnl.to_csv(out / "market_pnl.csv", index=False)
    if len(fills):
        fills.to_parquet(out / "fills.parquet", index=False)
    if len(paths):
        paths.to_parquet(out / "paths.parquet", index=False)

    grid.to_parquet(out / "grid.parquet", index=False)
    lagged.to_parquet(out / "predictions.parquet", index=False)
    spot.to_parquet(out / "spot.parquet", index=False)

    test_lagged = lagged.loc[lagged["split"] == "test"]
    example = test_lagged["slug"].iloc[0] if len(test_lagged) else frame["slug"].iloc[0]
    view = grid.merge(lagged[["slug", "t", *SCORED]], on=["slug", "t"], how="left")
    plot_calibration(test_lagged, SCORED, out / "plots" / "calibration.png")
    plot_example(view, paths, example, out / "plots" / "example_market.png")
    plot_cumulative(paths, out / "plots" / "cumulative_pnl.png")

    (out / "results.md").write_text(render_results(cfg, models, scores, metrics, example))
    _print_headline(scores, metrics)
    print(f"wrote {out}")


def tune_half_life(grid, spot, delta_s, candidates):
    rows = []
    best_half, best_score = None, None
    digital = D1()
    for half in candidates:
        scored = attach_sigma(grid, spot, float(half))
        scored["p"] = digital.predict(scored).to_numpy()
        scored["p"] = scored.groupby("slug", sort=False)["p"].shift(delta_s)
        val = scored.loc[(scored["split"] == "val") & scored["p"].notna()]
        score = brier_score(val["y"], val["p"]) if len(val) else float("inf")
        rows.append({"half_life_s": float(half), "val_brier": score, "n": int(len(val))})
        if best_score is None or score < best_score:
            best_score = score
            best_half = float(half)
            print(f"  half-life {half:g}s  validation Brier {score:.5f}")
    return best_half, pd.DataFrame(rows)


def fit_models(train, C, clip):
    lo, hi = clip
    models = {
        "G0": G0(lo, hi),
        "G1": G1(C=C, clip_lo=lo, clip_hi=hi),
        "D1": D1(lo, hi),
        "D2": D2(C=C, clip_lo=lo, clip_hi=hi),
        "market": MarketMid(lo, hi),
    }
    for model in models.values():
        model.fit(train)
    return models


def with_predictions(grid, models):
    frame = grid.copy()
    for name, model in models.items():
        frame[name] = model.predict(frame).to_numpy()
        if name in QUOTED:
            frame[f"dp_{name}"] = model.dp_dS(frame)
    return frame


def tune_quotes(val_grid, book_by, trades_by, delta, quote_cfg):
    needed = val_grid.copy()
    needed["p"] = needed["D1"]
    needed["dp_dS"] = needed["dp_D1"]
    rows = []
    best, best_pnl = None, None
    combos = list(product(
        quote_cfg["gamma_grid"],
        quote_cfg["kappa_grid"],
        quote_cfg["h_min_grid"],
        quote_cfg["tau_stop_grid"],
    ))
    for gamma, kappa, h_min, tau_stop in combos:
        quoter = _quoter(quote_cfg, delta, gamma, kappa, h_min, tau_stop)
        _, paths = replay(
            needed, None, None, quoter, "cautious", delta,
            book_by=book_by, trades_by=trades_by,
        )
        pnl = float(paths.loc[paths["settled"].astype(bool), "equity"].sum()) if len(paths) else 0.0
        rows.append({
            "gamma": float(gamma), "kappa": float(kappa), "h_min": float(h_min),
            "tau_stop": int(tau_stop), "val_pnl": pnl,
        })
        if best_pnl is None or pnl > best_pnl:
            best_pnl = pnl
            best = {
                "gamma": round(float(gamma), 6),
                "kappa": round(float(kappa), 6),
                "h_min": round(float(h_min), 4),
                "tau_stop": int(tau_stop),
            }
    print(f"  best validation PnL {best_pnl:.2f} at {best}")
    return best, pd.DataFrame(rows).sort_values("val_pnl", ascending=False)


def run_replays(frame, book_by, trades_by, delta, quote_cfg):
    quoter = _quoter(
        quote_cfg, delta,
        quote_cfg["gamma"], quote_cfg["kappa"], quote_cfg["h_min"], quote_cfg["tau_stop"],
    )
    fill_frames, path_frames = [], []
    for name in QUOTED:
        piece = frame.copy()
        piece["p"] = piece[name]
        piece["dp_dS"] = piece[f"dp_{name}"]
        for rule in ("cautious", "generous"):
            fills, paths = replay(
                piece, None, None, quoter, rule, delta,
                book_by=book_by, trades_by=trades_by,
            )
            if len(fills):
                fills = fills.copy()
                fills["model"] = name
                fills["fill_rule"] = rule
                fill_frames.append(fills)
            if len(paths):
                paths = paths.copy()
                paths["model"] = name
                paths["fill_rule"] = rule
                path_frames.append(paths)
    fills = pd.concat(fill_frames, ignore_index=True) if fill_frames else pd.DataFrame()
    paths = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    return fills, paths


def _quoter(quote_cfg, delta, gamma, kappa, h_min, tau_stop):
    return Quoter(
        h_min=h_min, kappa=kappa, gamma=gamma,
        size=quote_cfg["size"], inv_cap=quote_cfg["inv_cap"],
        tau_stop=tau_stop, delta_s=delta,
    )


def render_results(cfg, models, scores, metrics, example):
    lines = [
        "# Backtest results",
        "",
        "Parameters below were chosen on the validation day (2026-07-12) and then frozen. "
        "The test day (2026-07-13) was scored after that.",
        "",
        f"- EWMA half-life: {cfg['vol']['half_life_s']:g} seconds",
        f"- latency: {cfg['latency']['delta_s']} second",
        f"- gamma: {cfg['quote']['gamma']}",
        f"- kappa: {cfg['quote']['kappa']}",
        f"- h_min: {cfg['quote']['h_min']}",
        f"- tau_stop: {cfg['quote']['tau_stop']} seconds",
        f"- quote size: {cfg['quote']['size']}, inventory cap: {cfg['quote']['inv_cap']}",
        "",
        "G1 and D2 are fit on every train-market second, so each market's label is repeated about "
        "300 times. With five training markets, treat the fitted weights as a pilot. "
        "CVaR is the mean of the worst ceil(20% x n) per-market PnLs.",
        "",
        "## Forecast scores",
        "",
        _markdown_table(scores.loc[scores["tau_bucket"] == "all",
                                   ["split", "model", "n", "brier", "log_loss"]]),
        "",
        "Bucketed scores are in `forecast_scores.csv`.",
        "",
        "## Market maker",
        "",
        _markdown_table(metrics),
        "",
        f"The example tape is `{example}` (D1 quotes, cautious fills).",
        "",
        "## Fitted coefficients",
        "",
        "G1 coefficients are on standardized features.",
        "",
        _coef_list(models["G1"].coefficients()),
        "",
        "D2 coefficients are on standardized features. The base digital is still D1; "
        "these weights only recalibrate it.",
        "",
        _coef_list(models["D2"].coefficients()),
        "",
        f"G0 base rate: {models['G0'].p:.3f}",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame):
    if frame is None or len(frame) == 0:
        return "_none_"
    show = frame.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _coef_list(coefs):
    return "\n".join(f"- `{name}`: {value:.4f}" for name, value in coefs.items())


def _print_headline(scores, metrics):
    test_scores = scores.loc[(scores["split"] == "test") & (scores["tau_bucket"] == "all")]
    print("test Brier:")
    for row in test_scores.itertuples(index=False):
        print(f"  {row.model:8} {row.brier:.4f}")
    test_pnl = metrics.loc[metrics["split"] == "test"]
    print("test PnL:")
    for row in test_pnl.itertuples(index=False):
        print(f"  {row.model:8} {row.fill_rule:9} {row.pnl:8.2f}  drawdown {row.max_drawdown:.2f}")


def _write_config(path, cfg):
    text = yaml.safe_dump(_plain(cfg), sort_keys=False)
    path.write_text("# Tuned on the validation day, then frozen before the test day was scored.\n" + text)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return int(value) if value.is_integer() else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
