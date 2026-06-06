from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ingestion.EGX30_Client import EGXDataClient
import time

failed_tickers = ["RMDA"]  # Add any tickers that failed in the initial download

output_dir = Path("data") / "OHLCV"
client = EGXDataClient()

print(f"Retrying {len(failed_tickers)} failed stocks with longer delays...")
print("="*60)

for i, ticker in enumerate(failed_tickers, 1):
    print(f"\n[{i}/{len(failed_tickers)}] Retrying {ticker}...")
    
    if i > 1:
        print(f"Waiting 10s...")  # Longer delay
        time.sleep(10)
    
    df = client.fetch_stock_prices(ticker, n_bars=3765)
    
    if df is not None and not df.empty:
        output_file = output_dir / f"{ticker}.csv"
        df.to_csv(output_file)
        print(f"  ✓ SUCCESS! Saved {len(df)} rows")
    else:
        print(f"  ✗ Still failed")

print("\n" + "="*60)
print("Retry complete. Check results above.")