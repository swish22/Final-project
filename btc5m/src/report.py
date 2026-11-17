"""Write results.md from the test-stage tables."""
from __future__ import annotations

import pandas as pd

from mm.replay import FEE_RATE, REBATE_SHARE

MODEL_NOTES = {
    "G0": "share of training markets that resolved Up",
    "market": "the Polymarket mid itself",
    "G1": "logistic regression on Polymarket-only features",
    "D1": "BTC digital-option formula",
    "D2": "D1 recalibrated with short-horizon BTC features",
    "D3": "D2 plus the Polymarket features",
}
STRATEGY_NOTES = {
    "market": "Polymarket mid", "G1": "Generic model (G1)",
    "D1": "BTC model, active", "D1sel": "BTC model, selective",
}
MONEY = {"pnl": "{:,.0f}", "pnl_per_market": "{:+.2f}", "ci_lo": "{:+.2f}", "ci_hi": "{:+.2f}",
         "t_stat": "{:.2f}", "win_rate": "{:.0%}", "shares": "{:,.0f}", "max_drawdown": "{:,.0f}",
         "cvar": "{:.1f}", "worst_market": "{:.1f}", "rebate_est": "{:,.0f}", "pnl_sd": "{:.1f}"}
CENTS = {"spread_capture_c": "{:+.2f}", "markout_5s_c": "{:+.2f}", "markout_30s_c": "{:+.2f}",
         "pnl_per_share_c": "{:+.2f}", "quoted_spread_c": "{:.1f}", "uptime": "{:.0%}",
         "cross_fill_share": "{:.0%}", "mean_abs_inv": "{:.1f}", "shares": "{:,.0f}"}


def table(frame, columns, formats=None):
    """Markdown table. `columns` maps a frame column to its header."""
    if frame is None or len(frame) == 0:
        return "_none_"
    formats = formats or {}
    show = frame[list(columns)].copy()
    for col in show.columns:
        fmt = formats.get(col)
        if fmt:
            # Format, then drop the sign from a rounded zero so the table never shows "-0.00".
            show[col] = show[col].map(lambda v, f=fmt: "" if pd.isna(v) else _unsigned_zero(f.format(v)))
        elif pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    header = "| " + " | ".join(columns.values()) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _unsigned_zero(text):
    return text.lstrip("+-") if text.strip("+-0.,%") == "" else text


def verdict(lo, hi, lower_is_better):
    """Read a 95% interval for a difference against a baseline."""
    if lo > 0:
        return "worse" if lower_is_better else "better"
    if hi < 0:
        return "better" if lower_is_better else "worse"
    return "no clear difference"


def scenario_name(delta):
    return "current-second data (latency 0 s)" if delta == 0 else f"data {delta} s old (latency {delta} s)"


def render_results(cfg, kept, models, basis, scores, diffs, ece, metrics, pnl_diffs, lead, example):
    split = cfg["split"]
    counts = kept["split"].value_counts()
    q = cfg["quote"]
    tuned = sorted(int(d) for d in cfg["latency"]["scenarios"])
    learn = "train and validation days" if cfg.get("refit_on_learning_days", True) else "train days"
    by_lag = lead.set_index("lag_s")["corr"]
    alpha = cfg["risk"]["cvar_alpha"]
    floor = q.get("min_shares_per_market", 0)
    variants = q.get("variants") or {}

    def named(series):
        return series.map(STRATEGY_NOTES).fillna(series)

    metrics = metrics.assign(model=named(metrics["model"]))

    lines = [
        "# Backtest results",
        "",
        "## Setup",
        "",
        f"- Train: days {split['train'][0]}-{split['train'][1]} ({counts.get('train', 0)} markets). "
        f"Validation: days {split['val'][0]}-{split['val'][1]} ({counts.get('val', 0)} markets). "
        f"Test: days {split['test'][0]}-{split['test'][1]} ({counts.get('test', 0)} markets).",
        "- A day is a 24-hour block counted from the first recorded market (2026-07-10 23:20 UTC).",
        f"- Every setting below was chosen on the validation days and frozen. The models were then "
        f"refitted on the {learn} and the test days were scored once.",
        "- Two reaction speeds are reported side by side, each with its own tuned quote settings: "
        + " and ".join(scenario_name(d) for d in tuned) + ". A quote is refreshed once per second.",
        "",
        "## Frozen settings",
        "",
        f"- BTC proxy: {cfg['spot']['proxy']}",
        f"- Volatility: EWMA of 1-second returns, half-life {cfg['vol']['half_life_s']:g} s, "
        f"scaled by {cfg['vol']['sigma_scale']:g}",
        (f"- Settlement basis correction: on. {basis['beta']:.0%} of the gap between the strike and the "
         f"proxy at the open is assumed to persist to the close; the rest is noise of about "
         f"${basis['sd']:.2f}." if cfg["basis"]["enabled"] else "- Settlement basis correction: off"),
        f"- Quote size {q['size']} shares per side, inventory cap {q['inv_cap']} shares",
        f"- Each strategy's quote settings are the ones with the best Sharpe ratio on the validation days "
        f"among settings that traded at least {floor:g} shares per market there. Without a floor the best "
        "setting is always the one that barely quotes.",
    ]
    for name, variant in variants.items():
        lines.append(
            f"- {STRATEGY_NOTES.get(name, name)} uses the same forecasts as {variant['model']} with the floor "
            f"lowered to {variant.get('min_shares_per_market', floor):g} shares per market."
        )
    lines.append("")
    for delta in tuned:
        settings = pd.DataFrame([{"model": k, **v} for k, v in q["params"][str(delta)].items()])
        settings["model"] = named(settings["model"])
        lines += [
            f"Quote settings for {scenario_name(delta)}:",
            "",
            table(settings, {"model": "model", "h_min": "min half-spread", "kappa": "kappa",
                             "gamma": "gamma", "tau_stop": "stop quoting (s left)"},
                  {"h_min": "{:g}", "kappa": "{:g}", "gamma": "{:g}", "tau_stop": "{:g}"}),
            "",
        ]

    lines += [
        "## Forecast accuracy on the test days",
        "",
        "Every model is scored on the same market-seconds. Lower is better in every column.",
        "",
    ]
    for delta in reversed(tuned):
        overall = scores.loc[(scores["tau_bucket"] == "all") & (scores["latency_s"] == delta)]
        overall = overall.merge(ece.loc[ece["latency_s"] == delta, ["model", "ece"]], on="model", how="left")
        overall = overall.assign(what=overall["model"].map(MODEL_NOTES)).sort_values("brier")
        d = diffs.loc[diffs["latency_s"] == delta]
        d = d.assign(verdict=[verdict(lo, hi, True) for lo, hi in zip(d["ci_lo"], d["ci_hi"])])
        lines += [
            f"### With {scenario_name(delta)}",
            "",
            table(overall, {"model": "model", "what": "what it is", "brier": "Brier", "log_loss": "log loss",
                            "ece": "calibration error"}),
            "",
            "Brier score minus the baseline's. Negative means more accurate than the baseline. "
            "The interval is a 95% bootstrap that resamples whole hours.",
            "",
            table(d, {"model": "model", "baseline": "baseline", "brier_diff": "Brier difference",
                      "ci_lo": "low", "ci_hi": "high", "verdict": "than the baseline"},
                  {"brier_diff": "{:+.4f}", "ci_lo": "{:+.4f}", "ci_hi": "{:+.4f}"}),
            "",
        ]
    lines += [
        "Scores by time left are in `forecast_scores.csv` and `plots/brier_by_time_left.png`.",
        "",
        "## Does BTC move first?",
        "",
        "Correlation between the one-second change in the Polymarket mid and the one-second change "
        "in the D1 probability, in log-odds:",
        "",
        f"- same second: {by_lag.get(0, float('nan')):.2f}",
        f"- D1 one second earlier (BTC leads): {by_lag.get(1, float('nan')):.2f}",
        f"- D1 two seconds earlier: {by_lag.get(2, float('nan')):.2f}",
        f"- D1 one second later (Polymarket leads): {by_lag.get(-1, float('nan')):.2f}",
        "",
        "See `plots/lead_lag.png`.",
        "",
        "## Market making on the test days",
        "",
        "PnL is in dollars at settlement, with no fees (makers pay none) and no maker rebates. "
        "The interval is a 95% bootstrap that resamples whole hours.",
        "",
    ]
    main = metrics.loc[metrics["tuned"]]
    for delta in tuned:
        m = main.loc[main["latency_s"] == delta]
        cautious = m.loc[m["fill_rule"] == "cautious"].sort_values("pnl_per_market", ascending=False)
        generous = m.loc[m["fill_rule"] == "generous"].sort_values("pnl_per_market", ascending=False)
        pd_ = pnl_diffs.loc[pnl_diffs["a"].str.endswith(f"|{delta}")] if len(pnl_diffs) else pnl_diffs
        if len(pd_):
            pd_ = pd_.assign(a=named(pd_["a"].str.split("|").str[0]), b=named(pd_["b"].str.split("|").str[0]),
                             verdict=[verdict(lo, hi, False) for lo, hi in zip(pd_["ci_lo"], pd_["ci_hi"])])
        lines += [
            f"### With {scenario_name(delta)}",
            "",
            "Cautious fills:",
            "",
            table(cautious, {"model": "model", "pnl": "total PnL", "pnl_per_market": "PnL per market",
                             "ci_lo": "low", "ci_hi": "high", "t_stat": "t", "pnl_sd": "sd per market",
                             "win_rate": "markets with a profit", "max_drawdown": "max drawdown",
                             "cvar": f"CVaR {alpha:.0%}", "worst_market": "worst market"}, MONEY),
            "",
            "Where the money came from, in cents per share traded. Spread capture is the distance "
            "from the mid to our price at the fill. The next two columns are what that fill was worth "
            "5 and 30 seconds later, and the last is what it was worth at settlement.",
            "",
            table(cautious, {"model": "model", "shares": "shares", "quoted_spread_c": "quoted spread",
                             "uptime": "time quoting", "cross_fill_share": "fills from being run over",
                             "spread_capture_c": "spread capture", "markout_5s_c": "after 5 s",
                             "markout_30s_c": "after 30 s", "pnl_per_share_c": "at settlement",
                             "mean_abs_inv": "avg inventory"}, CENTS),
            "",
            "PnL per market minus the baseline's, on the same markets:",
            "",
            table(pd_, {"a": "model", "b": "baseline", "pnl_diff_per_market": "PnL difference",
                        "ci_lo": "low", "ci_hi": "high", "verdict": "than the baseline"},
                  {"pnl_diff_per_market": "{:+.2f}", "ci_lo": "{:+.2f}", "ci_hi": "{:+.2f}"}),
            "",
            "Generous fills (a shrink in displayed size that trades do not explain counts as cancels "
            "ahead of us):",
            "",
            table(generous, {"model": "model", "pnl": "total PnL", "pnl_per_market": "PnL per market",
                             "ci_lo": "low", "ci_hi": "high", "shares": "shares"}, MONEY),
            "",
        ]

    slow = metrics.loc[~metrics["tuned"]].sort_values(["model", "latency_s"])
    rebate = main.loc[main["fill_rule"] == "cautious"].sort_values(["latency_s", "model"])
    lines += [
        "## Robustness",
        "",
        f"Slower reaction, cautious fills, with the settings tuned for {tuned[-1]} s:",
        "",
        table(slow, {"model": "model", "latency_s": "latency (s)", "pnl": "total PnL",
                     "pnl_per_market": "PnL per market", "ci_lo": "low", "ci_hi": "high"}, MONEY),
        "",
        f"Maker rebates are left out of every number above. Polymarket returns {REBATE_SHARE:.0%} of crypto "
        f"taker fees (rate {FEE_RATE}) to makers in proportion to their filled volume. If our fills earned "
        "the average rebate, it would add:",
        "",
        table(rebate, {"model": "model", "latency_s": "latency (s)", "pnl": "total PnL",
                       "rebate_est": "estimated rebate ($)"}, MONEY),
        "",
        "## Fitted coefficients (standardized features)",
        "",
    ]
    for name in ("G1", "D2", "D3"):
        if name in models and hasattr(models[name], "coefficients"):
            coefs = ", ".join(f"`{k}` {v:+.3f}" for k, v in models[name].coefficients().items())
            lines += [f"- {name}: {coefs}"]
    if "G0" in models:
        lines += [f"- G0 base rate: {models['G0'].p:.3f}"]
    lines += ["", f"The example tape in `plots/example_market.png` is `{example}`.", ""]
    return "\n".join(lines)
