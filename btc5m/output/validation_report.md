# Validation report

Kept 14 of 15 markets. Dropped markets are excluded from the grid, the fit, and the replay.

Binance proxy winner matches the official winner in 14/14 kept markets (100%).

A market is dropped when the capture is incomplete, the clock-offset IQR exceeds 50 ms, the strike or the winner is missing, the book or the 1-second BTC path does not cover `[start, end]`, there is no spot mid at the open, or the Up bid/ask is missing on more than 50% of market seconds.

## Dropped

| slug | day | reasons |
| --- | --- | --- |
| btc-updown-5m-1783901400 | 2026-07-13 | missing_touch:55% |

## Kept

| slug | day | winner | proxy agrees | offset IQR ms |
| --- | --- | --- | --- | --- |
| btc-updown-5m-1783728000 | 2026-07-11 | down | True | 1.64 |
| btc-updown-5m-1783728300 | 2026-07-11 | down | True | 1.77 |
| btc-updown-5m-1783728600 | 2026-07-11 | down | True | 1.24 |
| btc-updown-5m-1783728900 | 2026-07-11 | down | True | 1.46 |
| btc-updown-5m-1783729200 | 2026-07-11 | up | True | 1.32 |
| btc-updown-5m-1783814400 | 2026-07-12 | up | True | 1.60 |
| btc-updown-5m-1783814700 | 2026-07-12 | down | True | 1.52 |
| btc-updown-5m-1783815000 | 2026-07-12 | down | True | 1.75 |
| btc-updown-5m-1783815300 | 2026-07-12 | up | True | 1.30 |
| btc-updown-5m-1783815600 | 2026-07-12 | down | True | 1.80 |
| btc-updown-5m-1783900800 | 2026-07-13 | up | True | 1.91 |
| btc-updown-5m-1783901100 | 2026-07-13 | up | True | 2.50 |
| btc-updown-5m-1783901700 | 2026-07-13 | down | True | 4.56 |
| btc-updown-5m-1783902000 | 2026-07-13 | down | True | 1.59 |

