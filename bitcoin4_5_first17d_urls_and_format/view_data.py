import pandas as pd
from read_capture import read_capture

magic, version, header, frames = read_capture("btc-updown-5m-1783725900.bin")

print(header["slug"], header["start_unix"], header["end_unix"])

# each frame is (wall_ns, event_number, dict_of_fields)
trades = [(t, f) for t, e, f in frames if e == 3]   # Polymarket trades
books  = [(t, f) for t, e, f in frames if e == 2]   # order book updates
result = [f for t, e, f in frames if e == 6]        # winner, open, close

print(len(trades), "trades;", len(books), "book updates")
print("outcome:", result)


trades_df = pd.DataFrame([{"wall_ns": t, **f} for t, e, f in frames if e == 3])