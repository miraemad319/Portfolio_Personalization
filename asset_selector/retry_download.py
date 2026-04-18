import sys
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # asset_selector/
sys.path.insert(0, str(_HERE))

from src.ingestion.egxlytics_client import EGXDataClient
import time

failed_tickers = ["MNHD", "PIOH", "GBAU", "PORT"]  # old rebrand tickers needed for merge history

output_dir = _HERE.parent / "data" / "raw" / "prices"
client = EGXDataClient()

print(f"Retrying {len(failed_tickers)} failed stocks with longer delays...")
print("="*60)

for i, ticker in enumerate(failed_tickers, 1):
    print(f"\n[{i}/{len(failed_tickers)}] Retrying {ticker}...")
    
    if i > 1:
        print(f"  ... Waiting 10s...")  # Longer delay
        time.sleep(20)
    
    df = client.fetch_stock_prices(ticker, n_bars=1500)
    
    if df is not None and not df.empty:
        output_file = output_dir / f"{ticker}.csv"
        df.to_csv(output_file)
        print(f"  OK SUCCESS! Saved {len(df)} rows")
    else:
        print(f"  FAIL Still failed")

print("\n" + "="*60)
print("Retry complete. Check results above.")

# Final disk check 
config_path = _HERE / "assets_egx30.json"
prices_dir  = _HERE.parent / "data" / "raw" / "prices"

with open(config_path) as f:
    config = json.load(f)

expected = set(config["historical_tickers"])
on_disk  = {f.stem for f in prices_dir.glob("*.csv")}
missing  = sorted(expected - on_disk)
delisted = set(config.get("delisted_tickers", {}).keys())

print("\n" + "="*60)
print(f"DISK CHECK")
print(f"  On disk  : {len(on_disk)}")
print(f"  Expected : {len(expected)}")

if missing:
    actionable = sorted(set(missing) - delisted)
    known_bad  = sorted(set(missing) & delisted)
    if actionable:
        print(f"\n  FAIL Still missing ({len(actionable)}) — retry these:")
        for t in actionable: print(f"      {t}")
    if known_bad:
        print(f"\n  ! Known issues ({len(known_bad)}) — manual action needed:")
        for t in known_bad: print(f"      {t}")
else:
    print("\n  OK All tickers on disk")