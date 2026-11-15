# Backtest results

Parameters below were chosen on the validation day (2026-07-12) and then frozen. The test day (2026-07-13) was scored after that.

- EWMA half-life: 300 seconds
- latency: 1 second
- gamma: 0.002
- kappa: 2.0
- h_min: 0.02
- tau_stop: 30 seconds
- quote size: 10, inventory cap: 100

G1 and D2 are fit on every train-market second, so each market's label is repeated about 300 times. With five training markets, treat the fitted weights as a pilot. CVaR is the mean of the worst ceil(20% x n) per-market PnLs.

## Forecast scores

| split | model | n | brier | log_loss |
| --- | --- | --- | --- | --- |
| train | G0 | 1500 | 0.1600 | 0.5004 |
| train | G1 | 1500 | 0.1413 | 0.4305 |
| train | D1 | 1500 | 0.1881 | 0.5349 |
| train | D2 | 1500 | 0.0951 | 0.2990 |
| train | market | 1377 | 0.2286 | 0.6478 |
| val | G0 | 1500 | 0.2800 | 0.7777 |
| val | G1 | 1500 | 0.2146 | 0.6083 |
| val | D1 | 1500 | 0.2614 | 0.8298 |
| val | D2 | 1500 | 0.2490 | 0.8345 |
| val | market | 1361 | 0.2948 | 0.8788 |
| test | G0 | 1200 | 0.3400 | 0.9163 |
| test | G1 | 1200 | 0.2153 | 0.6272 |
| test | D1 | 1200 | 0.1902 | 0.5500 |
| test | D2 | 1200 | 0.3414 | 1.1189 |
| test | market | 1074 | 0.2284 | 0.6516 |

Bucketed scores are in `forecast_scores.csv`.

## Market maker

| split | model | fill_rule | n_markets | n_fills | shares | pnl | mean_market_pnl | max_drawdown | cvar | markout_5s | markout_30s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | D1 | cautious | 4 | 191 | 1184.4428 | 23.4180 | 5.8545 | 85.1951 | -31.2000 | -0.0212 | -0.0442 |
| test | D2 | cautious | 4 | 164 | 996.3300 | -5.6883 | -1.4221 | 140.6000 | -65.4000 | -0.0271 | -0.0273 |
| test | G1 | cautious | 4 | 217 | 1341.4838 | -3.5343 | -0.8836 | 137.3834 | -62.5334 | -0.0229 | -0.0074 |
| test | market | cautious | 4 | 623 | 3801.0694 | -97.1376 | -24.2844 | 97.3876 | -27.6180 | -0.0205 | -0.0225 |
| test | D1 | generous | 4 | 203 | 1254.4428 | 24.3180 | 6.0795 | 84.0951 | -31.2000 | -0.0198 | -0.0447 |
| test | D2 | generous | 4 | 187 | 1067.3801 | -5.4056 | -1.3514 | 141.9000 | -65.5000 | -0.0215 | -0.0250 |
| test | G1 | generous | 4 | 257 | 1585.2272 | -3.1253 | -0.7813 | 138.2834 | -61.9334 | -0.0193 | -0.0053 |
| test | market | generous | 4 | 649 | 4011.7237 | -92.7251 | -23.1813 | 92.9751 | -29.6484 | -0.0198 | -0.0194 |
| train | D1 | cautious | 5 | 249 | 1593.7292 | 8.3499 | 1.6700 | 54.0947 | -13.6234 | -0.0362 | -0.0574 |
| train | D2 | cautious | 5 | 330 | 1970.2190 | 161.6674 | 32.3335 | 75.5000 | 20.9487 | -0.0311 | -0.0341 |
| train | G1 | cautious | 5 | 243 | 1454.7182 | 77.6703 | 15.5341 | 75.5000 | -60.1277 | -0.0346 | -0.0300 |
| train | market | cautious | 5 | 469 | 2927.3932 | -109.4847 | -21.8969 | 109.4847 | -26.1439 | -0.0349 | -0.0382 |
| train | D1 | generous | 5 | 292 | 1944.6345 | 1.0171 | 0.2034 | 57.2329 | -16.7000 | -0.0322 | -0.0635 |
| train | D2 | generous | 5 | 392 | 2341.0776 | 156.1333 | 31.2267 | 75.5000 | 16.4747 | -0.0290 | -0.0295 |
| train | G1 | generous | 5 | 319 | 1948.2450 | 84.1546 | 16.8309 | 75.5000 | -59.7426 | -0.0253 | -0.0230 |
| train | market | generous | 5 | 513 | 3290.5045 | -102.7960 | -20.5592 | 102.7960 | -27.5167 | -0.0296 | -0.0345 |
| val | D1 | cautious | 5 | 357 | 2248.8808 | 47.2143 | 9.4429 | 51.8600 | -25.5637 | -0.0143 | -0.0055 |
| val | D2 | cautious | 5 | 200 | 1110.0000 | 25.8415 | 5.1683 | 104.6422 | -53.8000 | -0.0166 | -0.0143 |
| val | G1 | cautious | 5 | 170 | 987.0000 | 6.8135 | 1.3627 | 106.1865 | -53.6300 | -0.0186 | -0.0164 |
| val | market | cautious | 5 | 649 | 4097.8706 | -59.6200 | -11.9240 | 59.9700 | -27.8446 | -0.0125 | -0.0128 |
| val | D1 | generous | 5 | 377 | 2357.9819 | 51.0468 | 10.2094 | 51.4195 | -25.9637 | -0.0124 | -0.0038 |
| val | D2 | generous | 5 | 225 | 1283.0928 | 31.3154 | 6.2631 | 105.5875 | -53.8000 | -0.0134 | -0.0030 |
| val | G1 | generous | 5 | 200 | 1216.6377 | 11.8162 | 2.3632 | 103.5175 | -53.5300 | -0.0155 | -0.0026 |
| val | market | generous | 5 | 671 | 4247.6484 | -67.2232 | -13.4446 | 67.5732 | -25.5042 | -0.0129 | -0.0154 |

The example tape is `btc-updown-5m-1783900800` (D1 quotes, cautious fills).

## Fitted coefficients

G1 coefficients are on standardized features.

- `pm_mid`: -1.0576
- `last_trade`: 1.9337
- `mid_chg_30`: 0.1300
- `imbalance`: 0.3134
- `tau`: -0.0938

D2 coefficients are on standardized features. The base digital is still D1; these weights only recalibrate it.

- `logit_d1`: 1.4541
- `binance_signed_vol_10s`: -0.0684
- `cb_bn_gap`: 1.2902
- `z`: 0.9308
- `b30`: -0.0288
- `b60`: 0.5888
- `b120`: 0.4354

G0 base rate: 0.200
