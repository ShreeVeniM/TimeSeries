import pandas as pd
import yfinance as yf

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error in load_data: {e}")
        return None

def load_yfinance_data(ticker, start, end):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end)
        return df
    except Exception as e:
        print(f"Error in load_yfinance_data: {e}")
        return None
