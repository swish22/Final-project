#!/usr/bin/env python3
"""Reference reader for bitcoin4 (BT4CAP01) and bitcoin5 (BT5CAP01) capture files. Standard library only.

usage:
    python3 read_capture.py <file.bin>            # header + per-event counts + first frames of each type
    python3 read_capture.py <file.bin> --dump     # every frame, one line each

Layout (all integers and floats little-endian, no padding):
    file   = magic[8] | version u16 | header_len u32 | header_json[header_len] | frame*
    frame  = wall_ns u64 | event u8 | payload_len u16 | payload[payload_len]
The last frame of a version-2 file is event 255 whose payload is a u64 = number of frames before it.
"""
import json
import struct
import sys

EVENT_NAMES = {
    1: "SPOT", 2: "BOOK", 3: "TRADE", 4: "OPEN", 5: "ORACLE", 6: "RESOLUTION", 7: "BTC_TRADE",
    10: "KALSHI_BOOK", 11: "KALSHI_DELTA", 12: "KALSHI_TRADE", 13: "KALSHI_TICKER", 14: "KALSHI_BRTI",
    15: "OPEN_KALSHI", 16: "RESOLUTION_KALSHI", 255: "END",
}
VENUE = {0: "binance", 1: "coinbase"}
SIDE = {0: "unknown", 1: "buy", 2: "sell"}
WINNER = {0: "up", 1: "down", 254: "none", 255: "forfeit"}
KSIDE = {1: "yes", 2: "no"}
BOOK_FIELDS = ("up_bid1", "up_bid1_size", "up_bid2", "up_bid2_size", "up_ask1", "up_ask1_size", "up_ask2", "up_ask2_size",
               "down_bid1", "down_bid1_size", "down_bid2", "down_bid2_size", "down_ask1", "down_ask1_size", "down_ask2",
               "down_ask2_size")


def decode(event, p):
    """Payload bytes -> dict of named fields."""
    if event == 1:    # venue u8, source_ts f64 (unix seconds), bid f64, ask f64
        v, ts, bid, ask = struct.unpack("<B3d", p)
        return {"venue": VENUE.get(v, v), "source_ts": ts, "bid": bid, "ask": ask}
    if event == 2:    # 16 x f64; a missing level has price -1.0 and size 0.0
        return dict(zip(BOOK_FIELDS, struct.unpack("<16d", p)))
    if event == 3:    # is_up u8 (1 = Up token, 0 = Down token), side u8, price f64, size f64, source_ts f64
        up, side, price, size, ts = struct.unpack("<BB3d", p)
        return {"token": "up" if up else "down", "side": SIDE.get(side, side), "price": price, "size": size, "source_ts": ts}
    if event in (4, 15):   # price to beat / Kalshi strike f64 (NaN if unavailable)
        return {"price_to_beat": struct.unpack("<d", p)[0]}
    if event == 5:    # price f64, source_ts f64
        price, ts = struct.unpack("<2d", p)
        return {"price": price, "source_ts": ts}
    if event in (6, 16):   # winner u8, open f64, close f64 (NaN if unavailable)
        w, o, c = struct.unpack("<B2d", p)
        return {"winner": WINNER.get(w, w), "open": o, "close": c}
    if event == 7:    # venue u8, aggressor side u8, price f64, size f64 (BTC), source_ts f64
        v, side, price, size, ts = struct.unpack("<BB3d", p)
        return {"venue": VENUE.get(v, v), "aggressor": SIDE.get(side, side), "price": price, "size": size, "source_ts": ts}
    if event == 10:   # side u8, count u16, then count x (price f64, size f64)  -- variable length
        side, n = struct.unpack_from("<BH", p, 0)
        lv = struct.unpack_from(f"<{2 * n}d", p, 3)
        return {"side": KSIDE.get(side, side), "levels": [(lv[2 * i], lv[2 * i + 1]) for i in range(n)]}
    if event == 11:   # side u8, price f64 (dollars), delta f64 (contracts, signed), ts_ms f64
        side, price, delta, ts = struct.unpack("<B3d", p)
        return {"side": KSIDE.get(side, side), "price": price, "delta": delta, "ts_ms": ts}
    if event == 12:   # taker outcome side u8, taker book side u8, yes_price f64, count f64, ts_ms f64
        o, b, yp, cnt, ts = struct.unpack("<BB3d", p)
        return {"taker_outcome": KSIDE.get(o, o), "taker_book_side": SIDE.get(b, b), "yes_price": yp, "count": cnt, "ts_ms": ts}
    if event == 13:   # 8 x f64
        k = ("yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size", "last_price", "volume", "dollar_volume", "ts_ms")
        return dict(zip(k, struct.unpack("<8d", p)))
    if event == 14:   # value f64, ts_ms f64, has_avg u8, avg_60s f64 (0.0 when has_avg = 0)
        value, ts, has, avg = struct.unpack("<2dBd", p)
        return {"value": value, "ts_ms": ts, "avg_60s": avg if has else None}
    if event == 255:  # number of frames written before this one
        return {"frame_count": struct.unpack("<Q", p)[0]}
    return {"raw": p.hex()}


def read_capture(path):
    """Returns (magic, version, header dict, list of (wall_ns, event, fields))."""
    raw = open(path, "rb").read()
    magic = raw[:8].decode("ascii")
    version, hlen = struct.unpack_from("<HI", raw, 8)
    header = json.loads(raw[14:14 + hlen])
    off, frames = 14 + hlen, []
    while off < len(raw):
        wall_ns, event, plen = struct.unpack_from("<QBH", raw, off)
        off += 11
        frames.append((wall_ns, event, decode(event, raw[off:off + plen])))
        off += plen
    return magic, version, header, frames


if __name__ == "__main__":
    magic, version, header, frames = read_capture(sys.argv[1])
    print("magic", magic, "version", version)
    print("header", json.dumps(header))
    if "--dump" in sys.argv:
        for wall_ns, event, f in frames:
            print(wall_ns, EVENT_NAMES.get(event, event), json.dumps(f))
        sys.exit(0)
    counts, firsts = {}, {}
    for wall_ns, event, f in frames:
        counts[event] = counts.get(event, 0) + 1
        firsts.setdefault(event, []).append((wall_ns, f)) if counts[event] <= 2 else None
    t0, t1 = frames[0][0], frames[-1][0]
    print(f"frames {len(frames)}; first wall_ns {t0} ({t0 / 1e9 - header['start_unix']:+.1f} s vs start); "
          f"last wall_ns {t1} ({t1 / 1e9 - header['end_unix']:+.1f} s vs end)")
    for event in sorted(counts):
        print(f"  event {event:3d} {EVENT_NAMES.get(event, '?'):<18} x {counts[event]}")
        for wall_ns, f in firsts[event]:
            print(f"      {wall_ns} {json.dumps(f)}")
