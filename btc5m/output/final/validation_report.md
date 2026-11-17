# Validation report

Kept 4859 of 4895 markets. Dropped markets are excluded from the grid, the fit, and the replay.

A market is dropped when the capture is incomplete or does not span the market, the clock-offset IQR exceeds 50 ms, the strike or the winner is missing, the strike was recorded more than 60 s after the open, the book is entirely empty on more than 5% of seconds or silent for more than 60 s, or the BTC path has holes. Markets are not dropped for having a one-sided book.

Up won 49.8% of the kept markets.

Does the exchange price alone pick the official winner (close >= open)?

| proxy | agrees with the official winner |
| --- | --- |
| Coinbase | 96.5% |
| Binance | 95.4% |
| mean of the two | 97.4% |

Book statistics over the kept markets:

- median share of seconds with a one-sided book: 3.0%
- median longest silence in the book: 1.3 s
- median taker prints per market: 1956
- strike recorded a median 16.8 s after the open

Kept markets per day: 0: 286, 1: 285, 2: 285, 3: 282, 4: 288, 5: 285, 6: 288, 7: 286, 8: 288, 9: 286, 10: 278, 11: 288, 12: 287, 13: 285, 14: 286, 15: 288, 16: 288

## Dropped

| reason | markets |
| --- | --- |
| spot_coverage | 17 |
| clock_iqr | 17 |
| book_gap | 15 |
| book_coverage | 2 |
| no_book | 2 |
| late_K | 1 |
| capture_coverage | 1 |
| strike_mismatch | 1 |

| slug | day | reasons |
| --- | --- | --- |
| btc-updown-5m-1783725600 | 0 | late_K,capture_coverage,book_coverage,spot_coverage |
| btc-updown-5m-1783774200 | 0 | book_gap |
| btc-updown-5m-1783836300 | 1 | book_gap |
| btc-updown-5m-1783888500 | 1 | book_coverage |
| btc-updown-5m-1783937700 | 2 | no_book,book_gap |
| btc-updown-5m-1783964100 | 2 | clock_iqr,spot_coverage |
| btc-updown-5m-1783964400 | 2 | clock_iqr,spot_coverage |
| btc-updown-5m-1784039400 | 3 | clock_iqr,spot_coverage |
| btc-updown-5m-1784039700 | 3 | clock_iqr,spot_coverage |
| btc-updown-5m-1784040900 | 3 | clock_iqr,spot_coverage |
| btc-updown-5m-1784041200 | 3 | clock_iqr,spot_coverage |
| btc-updown-5m-1784049000 | 3 | clock_iqr,spot_coverage |
| btc-updown-5m-1784049300 | 3 | clock_iqr,spot_coverage |
| btc-updown-5m-1784174400 | 5 | book_gap |
| btc-updown-5m-1784174700 | 5 | book_gap |
| btc-updown-5m-1784205600 | 5 | strike_mismatch |
| btc-updown-5m-1784393400 | 7 | clock_iqr,spot_coverage |
| btc-updown-5m-1784393700 | 7 | clock_iqr,spot_coverage |
| btc-updown-5m-1784536500 | 9 | clock_iqr,spot_coverage |
| btc-updown-5m-1784536800 | 9 | clock_iqr,spot_coverage |
| btc-updown-5m-1784607000 | 10 | book_gap |
| btc-updown-5m-1784607300 | 10 | book_gap |
| btc-updown-5m-1784607600 | 10 | book_gap |
| btc-updown-5m-1784607900 | 10 | book_gap |
| btc-updown-5m-1784608200 | 10 | book_gap |
| btc-updown-5m-1784608500 | 10 | book_gap |
| btc-updown-5m-1784608800 | 10 | book_gap |
| btc-updown-5m-1784622600 | 10 | clock_iqr |
| btc-updown-5m-1784655900 | 10 | clock_iqr,spot_coverage |
| btc-updown-5m-1784656200 | 10 | clock_iqr,spot_coverage |
| btc-updown-5m-1784824500 | 12 | no_book |
| btc-updown-5m-1784865600 | 13 | book_gap |
| btc-updown-5m-1784865900 | 13 | book_gap |
| btc-updown-5m-1784867100 | 13 | book_gap |
| btc-updown-5m-1784942700 | 14 | clock_iqr,spot_coverage |
| btc-updown-5m-1784943000 | 14 | clock_iqr,spot_coverage |
