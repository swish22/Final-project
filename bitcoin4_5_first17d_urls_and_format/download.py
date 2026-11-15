"""Download a small batch of BTC 5-minute capture files.
 
usage:
    python3 download.py 2026-07-11 5      # first 5 markets that start on Jul 11 (UTC)
    python3 download.py 2026-07-11        # all 288 markets that day (5-18 GB!)
 
Files go to raw/<slug>.bin. Uses Python's own HTTPS client instead of curl
(macOS curl gets its connection reset by this file host). A file is only kept
if the download finished, so a re-run skips good files and retries failed ones.
"""
import csv
import os
import shutil
import sys
import time
import urllib.request
 
INDEX = "bitcoin4_first17d_index.csv"
OUT = "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
 
 
def fetch(url, dest, tries=3):
    """Stream url to dest; return None on success, else the last error message."""
    err = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            return None
        except Exception as e:                  # network errors, HTTP 403/404, timeouts
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 * attempt)
    return err
 
 
if len(sys.argv) < 2:
    sys.exit(__doc__)
day = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
os.makedirs(OUT, exist_ok=True)
 
rows = [r for r in csv.DictReader(open(INDEX)) if r["start_utc"].startswith(day)]
if not rows:
    sys.exit(f"no markets start on {day}; data covers 2026-07-10 to 2026-07-27")
rows = rows[:limit]
print(f"{len(rows)} markets on {day}")
 
failed = []
for i, r in enumerate(rows, 1):
    path = os.path.join(OUT, r["slug"] + ".bin")
    if os.path.exists(path):
        print(f"[{i}/{len(rows)}] {r['slug']} already here, skipping")
        continue
    tmp = path + ".part"                       # half-finished downloads never look complete
    print(f"[{i}/{len(rows)}] {r['slug']} ({r['start_utc']} UTC) ...", end=" ", flush=True)
    err = fetch(r["url"], tmp)
    if err is None and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)
        print(f"{os.path.getsize(path) / 1e6:.0f} MB")
    else:
        if os.path.exists(tmp):
            os.remove(tmp)
        failed.append(r["slug"])
        print("FAILED -", err or "empty file")
 
print(f"done: {len(rows) - len(failed)} ok, {len(failed)} failed")
if failed:
    print("failed:", ", ".join(failed))