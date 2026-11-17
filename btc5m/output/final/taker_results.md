# Taker strategy results

## The rule

Each second, if the BTC model's probability moved by at least 0.02 in the last second and the Polymarket mid has followed by less than 75% of that move, trade 10 shares in the direction of the move at the displayed price (buy at the ask or sell at the bid). Close the trade 2 seconds later at the opposite touch. No new trades in the last 30 seconds, on a one-sided book, or in the second the strike is first recorded.

## Assumptions

- Zero reaction time: the decision at second t uses data up to t and fills against the book displayed at t.
- No fees.
- The displayed size is available to us, and our trades do not move the market.

These were chosen to see whether the signal exists at all. The last table shows what each one is worth.

## Results under those assumptions

The settings were chosen on the train days only. Validation and test days played no part in choosing them, and the test days were scored once.

| days | markets | trades per market | total PnL ($) | $ per market | low | high | t | cents per share | trades won | days profitable | hours profitable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3137 | 8.9 | +8,180 | +2.61 | +2.44 | +2.77 | 31.1 | +3.11 | 63% | 11 of 11 | 100% |
| val | 860 | 11.4 | +2,366 | +2.75 | +2.47 | +3.03 | 19.7 | +2.56 | 61% | 3 of 3 | 99% |
| test | 862 | 7.1 | +1,351 | +1.57 | +1.32 | +1.82 | 11.7 | +2.32 | 54% | 3 of 3 | 95% |

The interval is a 95% bootstrap that resamples whole hours.

Risk on the same runs:

| days | sd per market ($) | markets with a profit | markets with a loss | worst market ($) | worst day ($) | largest position (shares) |
| --- | --- | --- | --- | --- | --- | --- |
| train | 3.02 | 87% | 12% | -12.4 | +295 | 20 |
| val | 3.23 | 88% | 12% | -7.7 | +760 | 20 |
| test | 2.33 | 79% | 18% | -7.9 | +295 | 20 |

PnL by day:

| days | day | PnL ($) |
| --- | --- | --- |
| train | 0 | +557 |
| train | 1 | +645 |
| train | 2 | +972 |
| train | 3 | +1,003 |
| train | 4 | +889 |
| train | 5 | +746 |
| train | 6 | +905 |
| train | 7 | +295 |
| train | 8 | +552 |
| train | 9 | +814 |
| train | 10 | +806 |
| val | 11 | +835 |
| val | 12 | +771 |
| val | 13 | +760 |
| test | 14 | +295 |
| test | 15 | +318 |
| test | 16 | +738 |

## What the assumptions are worth (test days)

| scenario | total PnL ($) | $ per market | low | high | cents per share | days profitable |
| --- | --- | --- | --- | --- | --- | --- |
| as assumed: no delay, no fees | +1,351 | +1.57 | +1.32 | +1.82 | +2.32 | 3 of 3 |
| order arrives 1 s late | -494 | -0.57 | -0.66 | -0.48 | -0.83 | 0 of 3 |
| order arrives 2 s late | -635 | -0.74 | -0.85 | -0.62 | -1.06 | 0 of 3 |
| with Polymarket's taker fee | -117 | -0.14 | -0.33 | +0.05 | -0.20 | 1 of 3 |
| 1 s late and with the taker fee | -1,990 | -2.31 | -2.52 | -2.11 | -3.33 | 0 of 3 |

The taker fee is Polymarket's formula for crypto markets, 0.07 x price x (1 - price) per share, charged on the way in and on the way out.
