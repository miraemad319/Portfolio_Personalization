from egxpy.download import get_OHLCV_data
import pandas as pd

class EGXDataClient:
    """Client to fetch EGX stock data using egxpy."""
    
    def fetch_stock_prices(self, ticker, n_bars=3765):
        """
        Fetch historical prices for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            n_bars: Number of bars to fetch (default 3765 days)
        
        Returns:
            pandas DataFrame with OHLCV data
        """
        try:
            df = get_OHLCV_data(
                symbol=ticker,
                exchange='EGX',
                interval='Daily',
                n_bars=n_bars
            )
            return df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
    
    def fetch_multiple_stocks(self, tickers, n_bars=3765):
        """Fetch data for multiple stocks."""
        all_data = {}
        for ticker in tickers:
            print(f"Fetching {ticker}...")
            df = self.fetch_stock_prices(ticker, n_bars)
            if df is not None and not df.empty:
                all_data[ticker] = df
        return all_data