import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import json
from src.ingestion.EGX30_Client import EGXDataClient

def load_stock_list(config_path):
    """Load stock tickers from config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['tickers']

def download_all_prices(n_bars=2735):
    """Download historical prices for all EGX30 stocks."""
    # Setup paths - resolve from this file's location
    script_dir = Path(__file__).resolve().parent  # src/ingestion/
    project_root = script_dir.parent.parent  # AI-Portfolio-Generation/

    output_dir = project_root / "data" / "OHLCV"
    config_path = project_root / "src" / "config" / "EGX30_Tickers.json"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup client
    client = EGXDataClient()
    
    # Load tickers
    tickers = load_stock_list(config_path)
    
    print(f"\nDownloading {n_bars} days of data for {len(tickers)} stocks")
    print("="*60)
    
    # Setup client
    client = EGXDataClient()
    
    # Track successes and failures
    successful = []
    failed = []
    
    # Download each stock
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        df = client.fetch_stock_prices(ticker, n_bars)
        
        if df is not None and not df.empty:
            output_file = output_dir / f"{ticker}.csv"
            df.to_csv(output_file)
            print(f"  ✓ Saved {len(df)} rows to {output_file.name}")
            successful.append(ticker)
        else:
            print(f"  ✗ No data for {ticker}")
            failed.append(ticker)
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY:")
    print(f"  ✓ Successful: {len(successful)}/{len(tickers)}")
    print(f"  ✗ Failed: {len(failed)}/{len(tickers)}")
    
    if failed:
        print(f"\nFailed tickers: {', '.join(failed)}")

if __name__ == '__main__':
    download_all_prices()