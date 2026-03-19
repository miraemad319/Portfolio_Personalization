import pandas as pd
import talib
import os

#TODO: Add support & resistance
# List of EGX 30 stock tickers
tickers = [
    'ABUK', 'ADIB', 'AMOC', 'ARCC', 'BTFH', 'CCAP', 'CIEB', 'COMI', 'EAST', 
    'EGAL', 'EMFD', 'ETEL', 'FWRY', 'GBCO', 'HRHO', 'ISPH', 'JUFO', 'MASR', 
    'MCQE', 'MFPC', 'ORAS', 'ORHD', 'ORWE', 'PHDC', 'RAYA', 'SKPC', 'TMGH', 
    'VLMR', 'VLMRA', 'RMDA']

# Base folder (src/data)
data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "data")
raw_folder = os.path.join(data_folder, "raw", "prices")
processed_folder = os.path.join(data_folder, "processed")

# Create processed folder if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

# Loop through each ticker
for ticker in tickers:
    file_path = os.path.join(raw_folder, f"{ticker}.csv")
    
    # Check if CSV exists
    if not os.path.isfile(file_path):
        print(f"CSV for {ticker} not found at {file_path}")
        continue
    
    # Load CSV
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df = df.sort_values('datetime')
    
    # Calculate Indicators
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    df['RSI_50'] = talib.RSI(df['close'], timeperiod=50) 
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['MA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['MA_200'] = talib.SMA(df['close'], timeperiod=200)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )

    # Exponential Moving Averages
    df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
    df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)
    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
    df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
    
    df['VOL_MA_20'] = df['volume'].rolling(20).mean()
    df['VOL_MA_50'] = df['volume'].rolling(50).mean()
    df['VOL_ratio_20'] = df['volume'] / df['VOL_MA_20']

    # Save to processed folder
    processed_file_path = os.path.join(processed_folder, f"{ticker}_indicators.csv")
    df.to_csv(processed_file_path, index=False)
    print(f"Indicators added and saved for {ticker}")
