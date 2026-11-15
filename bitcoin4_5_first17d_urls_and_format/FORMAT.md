# bitcoin4 / bitcoin5 capture data: first 17 days of URLs and the file format

## What is in this zip

| file | contents |
|---|---|
| `bitcoin4_first17d_urls.txt` | 4,895 URLs, one per line: BTC 5-minute markets starting 2026-07-10 23:20 UTC up to (not including) 2026-07-27 23:20 UTC |
| `bitcoin4_first17d_index.csv` | same URLs with `manifest_line, slug, start_unix, start_utc, end_unix, url` |
| `bitcoin5_first17d_urls.txt` | 1,632 URLs, one per line: BTC 15-minute markets starting 2026-07-14 06:30 UTC up to (not including) 2026-07-31 06:30 UTC |
| `bitcoin5_first17d_index.csv` | same URLs with the same columns |
| `read_capture.py` | reference decoder (Python standard library only) |
| `FORMAT.md` | this file |

"First 17 days" = 17 x 24 h counted from the first recorded market of each recorder. The URLs are lines 1-4895 of
bitcoin4 `data.txt` and lines 1-1632 of bitcoin5 `data15m.txt`, in the original (chronological) order.

Coverage inside the window: bitcoin5 has every 15-minute market (1,632 of 1,632). bitcoin4 has 4,895 of 4,896; the
only missing market is `btc-updown-5m-1783823100` (2026-07-12 02:25 UTC).

## One URL = one file = one market

Each URL returns one raw capture file (not an archive). Download with e.g.

    curl --fail --location --silent --show-error --output <slug>.bin "<URL>"

The URL does not contain the market name; the slug is inside the file header (and in the index CSV). The recorder's
own file name is `<slug>.bin`, where slug = `btc-updown-5m-<start_unix>` (bitcoin4) or `btc-updown-15m-<start_unix>`
(bitcoin5). Sampled files are 17-64 MB each.

A file covers the market from about 300 s before its start to about 30-60 s after its end (the first file after a
recorder start begins late). Because of that overlap, consecutive files repeat the shared feeds (spot quotes, BTC
trades, BRTI) for the overlapping seconds.

## Binary layout

All integers and floats are little-endian. There is no padding or alignment.

    file   = magic[8] | version u16 | header_len u32 | header_json[header_len] | frame, frame, ...
    frame  = wall_ns u64 | event u8 | payload_len u16 | payload[payload_len]

| field | bytes | value |
|---|---|---|
| magic | 8 | ASCII `BT4CAP01` (bitcoin4) or `BT5CAP01` (bitcoin5) |
| version | 2 | u16 = 2 |
| header_len | 4 | u32, length of the JSON header in bytes |
| header_json | header_len | UTF-8 JSON object, see below |
| frames | rest of file | 11-byte frame header + payload, repeated to end of file |

So the first frame starts at byte offset `14 + header_len`.

### Header JSON

    {"slug": "btc-updown-5m-1783725900",
     "up_token": "<Polymarket token id of the Up outcome, decimal string>",
     "down_token": "<Polymarket token id of the Down outcome, decimal string>",
     "start_unix": 1783725900,          market start, unix seconds
     "end_unix": 1783726200,            market end, unix seconds (start + 300 or + 900)
     "created_wall_ns": 1783725720835096280,   recorder clock when the file was created, unix nanoseconds
     "clock": "monotonic_projected_wall"}

### Frame header

| field | type | meaning |
|---|---|---|
| wall_ns | u64 | recorder time in unix nanoseconds: `created_wall_ns` + monotonic time elapsed since file creation. Strictly increasing within a file (ties are bumped by 1 ns). It is stamped when the frame is written, i.e. shortly after the message was received. |
| event | u8 | event type, table below |
| payload_len | u16 | payload size in bytes |

`wall_ns` is the recorder machine's clock, not the exchange's. In the four files I decoded it ran 13-20 s behind
the exchange timestamps carried in the payloads (`source_ts`, `ts_ms`), so compare the two before relying on absolute
time.

### Event types and payloads

`f64` = 8-byte IEEE double. Enumerations: venue 0 = Binance, 1 = Coinbase; side 0 = unknown, 1 = buy, 2 = sell;
Kalshi side 1 = yes, 2 = no.

| event | name | recorder | payload bytes | payload fields in order |
|---|---|---|---|---|
| 1 | SPOT | both | 25 | venue u8, source_ts f64, bid f64, ask f64 |
| 2 | BOOK | both | 128 | 16 x f64, see below |
| 3 | TRADE | both | 26 | is_up u8, side u8, price f64, size f64, source_ts f64 |
| 4 | OPEN | both | 8 | price_to_beat f64 |
| 5 | ORACLE | both | 16 | price f64, source_ts f64 |
| 6 | RESOLUTION | both | 17 | winner u8, open f64, close f64 |
| 7 | BTC_TRADE | both | 26 | venue u8, aggressor side u8, price f64, size f64, source_ts f64 |
| 10 | KALSHI_BOOK | bitcoin5 | 3 + 16 x count | side u8, count u16, then count x (price f64, size f64) |
| 11 | KALSHI_DELTA | bitcoin5 | 25 | side u8, price f64, delta f64, ts_ms f64 |
| 12 | KALSHI_TRADE | bitcoin5 | 26 | taker_outcome_side u8, taker_book_side u8, yes_price f64, count f64, ts_ms f64 |
| 13 | KALSHI_TICKER | bitcoin5 | 64 | yes_bid, yes_bid_size, yes_ask, yes_ask_size, last_price, volume, dollar_volume, ts_ms (8 x f64) |
| 14 | KALSHI_BRTI | bitcoin5 | 25 | value f64, ts_ms f64, has_avg u8, avg_60s f64 |
| 15 | OPEN_KALSHI | bitcoin5 | 8 | strike f64 |
| 16 | RESOLUTION_KALSHI | bitcoin5 | 17 | winner u8, open f64, close f64 |
| 255 | END | both | 8 | frame_count u64 |

Details per event:

- **1 SPOT** - best bid / ask of BTC spot. Binance: `btcusdt@bookTicker`; Coinbase: `BTC-USD` ticker channel.
  `source_ts` is the exchange time in unix seconds (fractional). For Binance it is 0.0 in every file I decoded (the
  spot bookTicker message has no event time); Coinbase carries a real timestamp.
- **2 BOOK** - top two price levels of both Polymarket tokens, 16 doubles in this order:
  `up_bid1, up_bid1_size, up_bid2, up_bid2_size, up_ask1, up_ask1_size, up_ask2, up_ask2_size,`
  `down_bid1, down_bid1_size, down_bid2, down_bid2_size, down_ask1, down_ask1_size, down_ask2, down_ask2_size`.
  Prices are in dollars (0.01-0.99), sizes in shares. A level that does not exist is price -1.0, size 0.0. BOOK
  frames start once both tokens have received a full book snapshot; after that one is written whenever the locally
  maintained book is updated (snapshot or price change), so many consecutive frames are identical when the change was
  deeper than level 2. Only these top-two levels are stored - deeper Polymarket depth is not in the file.
- **3 TRADE** - a Polymarket trade (`last_trade_price` message). `is_up` 1 = Up token, 0 = Down token; `side` is the
  side reported by Polymarket (the taker's side); price in dollars, size in shares; `source_ts` unix seconds.
- **4 OPEN** - written once at market start: the price to beat in USD (NaN if it could not be fetched).
- **5 ORACLE** - Chainlink BTC price from Polymarket's live-data feed: price USD, `source_ts` unix seconds. Defined in
  the format, but the four files I decoded contain no ORACLE frames.
- **6 RESOLUTION** - written once after the market ends: winner 0 = Up, 1 = Down, 255 = could not be resolved
  (open and close are then NaN); `open` and `close` are the opening and closing reference prices in USD.
- **7 BTC_TRADE** - BTC spot trades. Binance `btcusdt@aggTrade`: aggressor 1 = buyer was the taker, 2 = seller was
  the taker. Coinbase: last trade price / size / side from the ticker channel. Size in BTC, `source_ts` unix seconds.
- **10 KALSHI_BOOK** - full Kalshi order book snapshot for one side (variable length): `count` levels of
  (price in dollars, size in contracts). Snapshots come in pairs, one frame for side 1 (yes) and one for side 2 (no).
- **11 KALSHI_DELTA** - change to one Kalshi book level: price in dollars, `delta` = signed change in contracts,
  `ts_ms` = Kalshi timestamp in unix milliseconds. Apply deltas to the latest snapshot to rebuild the book.
- **12 KALSHI_TRADE** - `taker_outcome_side` 1 = yes, 2 = no; `taker_book_side` 1 = bid, 2 = ask, 0 = unknown;
  `yes_price` in dollars; `count` in contracts; `ts_ms` unix milliseconds.
- **13 KALSHI_TICKER** - Kalshi ticker: yes bid / size, yes ask / size, last price, cumulative volume (contracts),
  dollar volume, `ts_ms` unix milliseconds.
- **14 KALSHI_BRTI** - CF Benchmarks BRTI index value, `ts_ms` unix milliseconds, then `has_avg` (1 or 0) and the
  60-second average (0.0 when `has_avg` is 0).
- **15 OPEN_KALSHI** - the Kalshi market's strike in USD (NaN if unavailable).
- **16 RESOLUTION_KALSHI** - winner 0 = Up / yes, 1 = Down / no, 254 = unresolved; `open` = strike, `close` =
  settlement value in USD (NaN if unavailable).
- **255 END** - always the last frame of a complete file; `frame_count` = number of frames before it. A file
  without it is truncated.

In bitcoin5 files the Kalshi frames for the same 15-minute window are interleaved with the Polymarket frames in the
same file. bitcoin4 files contain only events 1-7 and 255.

## How this was checked

The layout above was taken from the recorder source (`format.rs`, `recorder.rs`, `feeds.rs`, `kalshi.rs`) and then
confirmed by decoding four real files end to end with `read_capture.py`: the first-day and last-day bitcoin4 files
and the first and last bitcoin5 files of this window. In all four the magic, version, header, payload sizes and the
END frame count matched. Which message triggers a BOOK frame is described from the current recorder source; I did
not check whether that detail was identical in July.
