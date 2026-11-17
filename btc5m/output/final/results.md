# Backtest results

## Setup

- Train: days 0-10 (3137 markets). Validation: days 11-13 (860 markets). Test: days 14-16 (862 markets).
- A day is a 24-hour block counted from the first recorded market (2026-07-10 23:20 UTC).
- Every setting below was chosen on the validation days and frozen. The models were then refitted on the train and validation days and the test days were scored once.
- Two reaction speeds are reported side by side, each with its own tuned quote settings: current-second data (latency 0 s) and data 1 s old (latency 1 s). A quote is refreshed once per second.

## Frozen settings

- BTC proxy: mean
- Volatility: EWMA of 1-second returns, half-life 1200 s, scaled by 1.5
- Settlement basis correction: on. 23% of the gap between the strike and the proxy at the open is assumed to persist to the close; the rest is noise of about $1.95.
- Quote size 10 shares per side, inventory cap 100 shares
- Each strategy's quote settings are the ones with the best Sharpe ratio on the validation days among settings that traded at least 50 shares per market there. Without a floor the best setting is always the one that barely quotes.
- BTC model, selective uses the same forecasts as D1 with the floor lowered to 5 shares per market.

Quote settings for current-second data (latency 0 s):

| model | min half-spread | kappa | gamma | stop quoting (s left) |
| --- | --- | --- | --- | --- |
| Polymarket mid | 0.05 | 0 | 0 | 120 |
| Generic model (G1) | 0.05 | 0 | 0 | 120 |
| BTC model, active | 0.05 | 2 | 0.00025 | 120 |
| BTC model, selective | 0.01 | 16 | 0.001 | 60 |

Quote settings for data 1 s old (latency 1 s):

| model | min half-spread | kappa | gamma | stop quoting (s left) |
| --- | --- | --- | --- | --- |
| Polymarket mid | 0.05 | 0 | 0 | 120 |
| Generic model (G1) | 0.05 | 0 | 0 | 120 |
| BTC model, active | 0.01 | 2 | 0 | 120 |
| BTC model, selective | 0.02 | 16 | 0.002 | 60 |

## Forecast accuracy on the test days

Every model is scored on the same market-seconds. Lower is better in every column.

### With data 1 s old (latency 1 s)

| model | what it is | Brier | log loss | calibration error |
| --- | --- | --- | --- | --- |
| D3 | D2 plus the Polymarket features | 0.1551 | 0.4618 | 0.0122 |
| G1 | logistic regression on Polymarket-only features | 0.1552 | 0.4621 | 0.0124 |
| market | the Polymarket mid itself | 0.1553 | 0.4628 | 0.0164 |
| D2 | D1 recalibrated with short-horizon BTC features | 0.1649 | 0.4938 | 0.0156 |
| D1 | BTC digital-option formula | 0.1657 | 0.4973 | 0.0285 |
| G0 | share of training markets that resolved Up | 0.2500 | 0.6932 | 0.0048 |

Brier score minus the baseline's. Negative means more accurate than the baseline. The interval is a 95% bootstrap that resamples whole hours.

| model | baseline | Brier difference | low | high | than the baseline |
| --- | --- | --- | --- | --- | --- |
| G0 | market | +0.0947 | +0.0869 | +0.1020 | worse |
| G1 | market | -0.0001 | -0.0004 | +0.0001 | no clear difference |
| D1 | market | +0.0104 | +0.0057 | +0.0162 | worse |
| D2 | market | +0.0097 | +0.0049 | +0.0158 | worse |
| D3 | market | -0.0002 | -0.0007 | +0.0004 | no clear difference |
| G0 | G1 | +0.0949 | +0.0869 | +0.1022 | worse |
| market | G1 | +0.0001 | -0.0001 | +0.0004 | no clear difference |
| D1 | G1 | +0.0105 | +0.0058 | +0.0165 | worse |
| D2 | G1 | +0.0098 | +0.0049 | +0.0158 | worse |
| D3 | G1 | 0.0000 | -0.0004 | +0.0004 | no clear difference |

### With current-second data (latency 0 s)

| model | what it is | Brier | log loss | calibration error |
| --- | --- | --- | --- | --- |
| D3 | D2 plus the Polymarket features | 0.1546 | 0.4604 | 0.0121 |
| G1 | logistic regression on Polymarket-only features | 0.1547 | 0.4607 | 0.0124 |
| market | the Polymarket mid itself | 0.1548 | 0.4614 | 0.0164 |
| D2 | D1 recalibrated with short-horizon BTC features | 0.1645 | 0.4924 | 0.0155 |
| D1 | BTC digital-option formula | 0.1652 | 0.4958 | 0.0284 |
| G0 | share of training markets that resolved Up | 0.2500 | 0.6932 | 0.0048 |

Brier score minus the baseline's. Negative means more accurate than the baseline. The interval is a 95% bootstrap that resamples whole hours.

| model | baseline | Brier difference | low | high | than the baseline |
| --- | --- | --- | --- | --- | --- |
| G0 | market | +0.0952 | +0.0874 | +0.1025 | worse |
| G1 | market | -0.0001 | -0.0004 | +0.0001 | no clear difference |
| D1 | market | +0.0104 | +0.0057 | +0.0162 | worse |
| D2 | market | +0.0097 | +0.0049 | +0.0158 | worse |
| D3 | market | -0.0002 | -0.0006 | +0.0004 | no clear difference |
| G0 | G1 | +0.0953 | +0.0874 | +0.1027 | worse |
| market | G1 | +0.0001 | -0.0001 | +0.0004 | no clear difference |
| D1 | G1 | +0.0105 | +0.0058 | +0.0164 | worse |
| D2 | G1 | +0.0098 | +0.0050 | +0.0158 | worse |
| D3 | G1 | 0.0000 | -0.0004 | +0.0004 | no clear difference |

Scores by time left are in `forecast_scores.csv` and `plots/brier_by_time_left.png`.

## Does BTC move first?

Correlation between the one-second change in the Polymarket mid and the one-second change in the D1 probability, in log-odds:

- same second: 0.40
- D1 one second earlier (BTC leads): 0.23
- D1 two seconds earlier: 0.03
- D1 one second later (Polymarket leads): 0.11

See `plots/lead_lag.png`.

## Market making on the test days

PnL is in dollars at settlement, with no fees (makers pay none) and no maker rebates. The interval is a 95% bootstrap that resamples whole hours.

### With current-second data (latency 0 s)

Cautious fills:

| model | total PnL | PnL per market | low | high | t | sd per market | markets with a profit | max drawdown | CVaR 5% | worst market |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC model, selective | 260 | +0.30 | -0.38 | +0.93 | 0.89 | 11.1 | 16% | 197 | -26.5 | -84.2 |
| BTC model, active | -565 | -0.66 | -3.43 | +1.91 | -0.50 | 36.4 | 44% | 2,030 | -61.7 | -102.5 |
| Generic model (G1) | -1,159 | -1.34 | -2.19 | -0.51 | -3.11 | 13.4 | 39% | 1,180 | -33.3 | -64.3 |
| Polymarket mid | -1,779 | -2.06 | -2.95 | -1.21 | -4.60 | 11.7 | 36% | 1,895 | -29.2 | -71.4 |

Where the money came from, in cents per share traded. Spread capture is the distance from the mid to our price at the fill. The next two columns are what that fill was worth 5 and 30 seconds later, and the last is what it was worth at settlement.

| model | shares | quoted spread | time quoting | fills from being run over | spread capture | after 5 s | after 30 s | at settlement | avg inventory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC model, selective | 12,317 | 67.3 | 80% | 15% | +1.32 | -0.17 | +1.78 | +2.11 | 4.2 |
| BTC model, active | 132,951 | 11.8 | 60% | 14% | +1.74 | -1.40 | -1.40 | -0.43 | 60.1 |
| Generic model (G1) | 58,896 | 10.9 | 60% | 14% | +5.14 | -2.35 | -2.34 | -1.97 | 16.7 |
| Polymarket mid | 63,573 | 10.9 | 60% | 13% | +5.44 | -2.50 | -2.69 | -2.80 | 15.9 |

PnL per market minus the baseline's, on the same markets:

| model | baseline | PnL difference | low | high | than the baseline |
| --- | --- | --- | --- | --- | --- |
| Generic model (G1) | Polymarket mid | +0.72 | -0.04 | +1.43 | no clear difference |
| BTC model, active | Polymarket mid | +1.41 | -1.20 | +3.83 | no clear difference |
| BTC model, active | Generic model (G1) | +0.69 | -2.01 | +3.21 | no clear difference |
| BTC model, selective | Polymarket mid | +2.37 | +1.21 | +3.50 | better |
| BTC model, selective | Generic model (G1) | +1.65 | +0.53 | +2.71 | better |

Generous fills (a shrink in displayed size that trades do not explain counts as cancels ahead of us):

| model | total PnL | PnL per market | low | high | shares |
| --- | --- | --- | --- | --- | --- |
| BTC model, selective | 255 | +0.30 | -0.39 | +0.92 | 13,106 |
| BTC model, active | -547 | -0.63 | -3.40 | +1.93 | 134,724 |
| Generic model (G1) | -1,159 | -1.35 | -2.19 | -0.52 | 58,966 |
| Polymarket mid | -1,780 | -2.06 | -2.96 | -1.21 | 63,643 |

### With data 1 s old (latency 1 s)

Cautious fills:

| model | total PnL | PnL per market | low | high | t | sd per market | markets with a profit | max drawdown | CVaR 5% | worst market |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC model, selective | -1 | 0.00 | -0.50 | +0.48 | 0.00 | 7.7 | 12% | 181 | -16.9 | -72.1 |
| BTC model, active | -1,011 | -1.17 | -3.45 | +1.19 | -1.01 | 33.5 | 41% | 1,882 | -59.0 | -103.2 |
| Generic model (G1) | -2,009 | -2.33 | -3.37 | -1.36 | -4.42 | 15.1 | 40% | 2,071 | -37.4 | -57.5 |
| Polymarket mid | -2,533 | -2.94 | -4.05 | -1.89 | -5.38 | 13.3 | 36% | 2,677 | -31.8 | -70.9 |

Where the money came from, in cents per share traded. Spread capture is the distance from the mid to our price at the fill. The next two columns are what that fill was worth 5 and 30 seconds later, and the last is what it was worth at settlement.

| model | shares | quoted spread | time quoting | fills from being run over | spread capture | after 5 s | after 30 s | at settlement | avg inventory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC model, selective | 8,071 | 79.6 | 80% | 18% | +1.95 | -0.06 | +1.84 | -0.01 | 2.2 |
| BTC model, active | 100,752 | 15.1 | 60% | 14% | +2.05 | -1.74 | -1.63 | -1.00 | 51.9 |
| Generic model (G1) | 85,258 | 10.9 | 60% | 13% | +3.58 | -2.26 | -2.30 | -2.36 | 20.8 |
| Polymarket mid | 85,673 | 10.9 | 60% | 13% | +3.59 | -2.30 | -2.50 | -2.96 | 19.8 |

PnL per market minus the baseline's, on the same markets:

| model | baseline | PnL difference | low | high | than the baseline |
| --- | --- | --- | --- | --- | --- |
| Generic model (G1) | Polymarket mid | +0.61 | -0.26 | +1.48 | no clear difference |
| BTC model, active | Polymarket mid | +1.77 | -0.22 | +3.77 | no clear difference |
| BTC model, active | Generic model (G1) | +1.16 | -1.05 | +3.37 | no clear difference |
| BTC model, selective | Polymarket mid | +2.94 | +1.73 | +4.18 | better |
| BTC model, selective | Generic model (G1) | +2.33 | +1.14 | +3.62 | better |

Generous fills (a shrink in displayed size that trades do not explain counts as cancels ahead of us):

| model | total PnL | PnL per market | low | high | shares |
| --- | --- | --- | --- | --- | --- |
| BTC model, selective | -10 | -0.01 | -0.52 | +0.47 | 8,805 |
| BTC model, active | -859 | -1.00 | -3.23 | +1.36 | 102,482 |
| Generic model (G1) | -2,010 | -2.33 | -3.37 | -1.36 | 85,328 |
| Polymarket mid | -2,534 | -2.94 | -4.05 | -1.89 | 85,733 |

## Robustness

Slower reaction, cautious fills, with the settings tuned for 1 s:

| model | latency (s) | total PnL | PnL per market | low | high |
| --- | --- | --- | --- | --- | --- |
| BTC model, active | 2 | -956 | -1.11 | -3.08 | +0.88 |
| BTC model, active | 5 | -674 | -0.78 | -2.28 | +0.71 |
| BTC model, selective | 2 | -67 | -0.08 | -0.55 | +0.38 |
| BTC model, selective | 5 | 14 | +0.02 | -0.35 | +0.40 |
| Generic model (G1) | 2 | -2,411 | -2.80 | -3.92 | -1.67 |
| Generic model (G1) | 5 | -2,178 | -2.53 | -3.70 | -1.46 |
| Polymarket mid | 2 | -2,719 | -3.15 | -4.09 | -2.20 |
| Polymarket mid | 5 | -2,333 | -2.71 | -3.55 | -1.92 |

Maker rebates are left out of every number above. Polymarket returns 20% of crypto taker fees (rate 0.07) to makers in proportion to their filled volume. If our fills earned the average rebate, it would add:

| model | latency (s) | total PnL | estimated rebate ($) |
| --- | --- | --- | --- |
| BTC model, active | 0 | -565 | 401 |
| BTC model, selective | 0 | 260 | 21 |
| Generic model (G1) | 0 | -1,159 | 182 |
| Polymarket mid | 0 | -1,779 | 197 |
| BTC model, active | 1 | -1,011 | 285 |
| BTC model, selective | 1 | -1 | 13 |
| Generic model (G1) | 1 | -2,009 | 263 |
| Polymarket mid | 1 | -2,533 | 263 |

## Fitted coefficients (standardized features)

- G1: `logit_mid` +1.095, `logit_last` +1.115, `mid_chg_30` -0.010, `imbalance` +0.050, `pm_flow_30` -0.041, `tau` +0.015
- D2: `logit_d1` -1.699, `z` +4.954, `binance_flow_10s` +0.032, `gap_z` +0.049, `b30` -0.004, `b60` -0.011, `b120` -0.018
- D3: `logit_d1` -0.214, `z` +0.477, `binance_flow_10s` +0.005, `gap_z` +0.037, `b30` +0.008, `b60` +0.017, `b120` +0.029, `logit_mid` +1.147, `logit_last` +0.926, `mid_chg_30` -0.007, `imbalance` +0.049, `pm_flow_30` -0.036
- G0 base rate: 0.498

The example tape in `plots/example_market.png` is `btc-updown-5m-1784994000`.
