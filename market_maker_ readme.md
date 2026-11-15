# BTC 5-Minute Market Maker 

## Project Overview
This project backtests two market makers on 4,895 Polymarket BTC 5-minute markets recorded between Jul 10–27, 2026 (`bitcoin4` files). 

**The Core Question:** Does a market maker that prices "Up" tokens using live BTC prices and settlement rules (Domain model) outperform one that only observes Polymarket's internal order book (Generic model), in terms of forecast accuracy, profit, and risk?

## System Architecture Pipeline
1. **Raw Data:** 4,895 `.bin` capture files (83–313 GB total).
2. **Converter:** Downloads, decodes, aligns clocks, dedupes, maps to the "Up" token, and writes to per-market Parquet tables.
3. **Validation:** Checks for complete files, stable clocks, strike consistency, proxy agreement, and base rates.
4. **Forecasting (P(Up)):** Generates probabilities every second using Base Rate (G0), Generic (G1), Digital (D1), and Recalibrated (D2) models.
5. **Market Maker & Replay:** Quotes around $P(\text{Up})$ with inventory skew and queue-aware fills on the recorded book. Scores via PnL, markout, drawdown, and CVaR.

---

## Repository Layout
```text
btc5m/
  index/               # bitcoin4_first17d_index.csv (input URLs and slugs)
  data/parquet/        # converter output (partitioned by day and table)
  src/
    convert/           # download.py, decode.py (wraps read_capture.py), align.py, grid.py
    validate/          # checks.py -> validation_report.md
    models/            # base.py, g0.py, g1.py, d1.py, d2.py
    mm/                # quoter.py, queue_fill.py, replay.py
    eval/              # forecast_scores.py, mm_metrics.py, plots.py
  config.yaml          # parameters: splits, latency, tau_stop, quote size, inv cap
  run_all.py           # Pipeline orchestrator
```

### Core Interfaces
*   **Forecaster:** A class with `fit(grid_train)` and `predict(grid) -> Series` of $P(\text{Up})$, one value per (market, second).
*   **Quoter:** A function `quote(p, sigma, tau, inventory, book) -> (bid, ask, size)`.
*   **Replay:** `replay(grid, book, trades, quoter, fill_rule) -> fills` returns one row per fill.

---

## Data Model & Schemas

### Binary Capture Format (`.bin`)
*   **1 SPOT:** Best bid/ask of BTC spot (Binance and Coinbase). *Note: Binance `source_ts` is always 0.0 in this dataset.*
*   **2 BOOK:** Top 2 price levels for Up/Down tokens (prices $0.01-$0.99).
*   **3 TRADE:** Polymarket trades (price, size, side).
*   **4 OPEN:** Strike price to beat ($K$).
*   **6 RESOLUTION:** Official open/close prices and winner.
*   **7 BTC_TRADE:** BTC spot trades (Binance/Coinbase).
*   **255 END:** Frame count to detect truncated files.

### Parquet Tables (Converter Output)
| Table | Granularity | Main Columns |
| :--- | :--- | :--- |
| `markets` | 1 row / market | slug, start, end, K (OPEN), res_open, res_close, winner, clock_offset, offset_iqr_ms, n_frames, flags |
| `book` | 1 row / book change | ts, up_bid1/size, up_bid2/size, up_ask1/size, up_ask2/size, plus Down levels |
| `trades` | 1 row / PM trade | ts, token, side, price, size, up_price, up_side, paired_flag |
| `btc` | 1 row / BTC quote | ts, venue, kind, bid, ask, price, size, aggressor |
| `grid` | 1 row / second | t, $\tau$ (seconds left), binance_mid, coinbase_mid, proxy $S_t$, $\sigma_t$, up_bid1, up_ask1, pm_mid, last_trade, flow features |

### Critical Data Quirks
*   **Clock Offset:** Recorder clock `wall_ns` ran 13–20s behind exchange time. Align using: `offset = median(source_ts - wall_ns/1e9)` on Binance `BTC_TRADE` frames. Apply to all frames: `ts = wall_ns/1e9 + offset`.
*   **Missing Oracle:** Chainlink `ORACLE` event is empty. Must proxy with Binance/Coinbase and check against `RESOLUTION`.
*   **Up Token Mapping:** Map everything to the Up token. A Down trade at price $y$ on side $s$ becomes an Up trade at $1 - y$ on the opposite side.

---

## Forecasting Models

Models output $P(\text{Up})$ for each market at each second.
*   **G0 (Base Rate):** Share of training markets resolving Up.
*   **G1 (Generic):** Logistic regression (or gradient boosting) on Polymarket features (mid, last trade, 30s change, imbalance, $\tau$).
*   **Market:** Polymarket mid at the same second.
*   **D1 (Digital):** Treats BTC as a driftless random walk. Clipped to [0.005, 0.995].
    $$P_t(\text{Up}) = \Phi\!\left(\frac{\ln\big((S_t + b)/K\big)}{\sigma_t\sqrt{\tau}}\right),\qquad b = K - S_{\text{open}}$$
*   **D2 (Recalibrated):** Logistic regression on `logit(D1)`, Binance order-flow imbalance (10s), Coinbase-Binance gap, distance to $K$ in $\sigma$ units, and $\tau$ bucket.

---

## Market Maker Rules

**Quoting Formula:**
$$\text{bid}=c-h,\quad \text{ask}=c+h,\qquad c = p_{t-\delta} - \gamma\, q,\qquad h = \max\!\big(h_{\min},\ \kappa\, \lvert\partial p/\partial S\rvert\, S\, \sigma\sqrt{\delta}\big)$$
Where $p$ is model's $P(\text{Up})$ delayed by latency $\delta$, $q$ is inventory, $\gamma$ is inventory skew, and $h$ is the half-spread (widened by BTC volatility).

**Execution & Fills:**
*   Round to 1¢ tick. *Never cross the recorded book.*
*   Quote 10 shares per side. Stop quoting if inventory hits cap or if time remaining $< \tau_{stop}$ (5-30s).
*   **Queue-Aware Fills:**
    *   *Better than best price:* First in line. Fill on taker flow.
    *   *Joining best price:* Wait for displayed size $Q$ to deplete. (Test both cautious/generous fill assumptions).
    *   *Worse than best price:* Fill only if trades print through the price.

---

## Development & Backtest Rules
1. **No Look-Ahead:** Every feature at second $t$ uses data stamped at or before $t - \delta$. The grid rows must strictly hold the *last known value*.
2. **Train/Val/Test Split:**
    *   **Train (Days 1–10):** Fit G1, D2.
    *   **Validate (Days 11–13):** Tune volatility half-life, parameters, and market maker settings.
    *   **Test (Days 14–17):** Touched *once* at the very end after `config.yaml` is committed.
3. **Data Integrity:** Markets failing validation checks (e.g., missing ends, high clock drift, strike mismatch) are dropped and counted, never silently fixed.