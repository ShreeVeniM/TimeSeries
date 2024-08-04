import pandas as pd


def preprocess_data(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.iloc[:-2, 0:2]
        df = df.set_index('Date')
        return df
    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        return None

def preprocess_yfinance_data(df):
    try:
        df['Next_day'] = df['Close'].shift(-1)
        df['Target'] = (df['Next_day'] > df['Close']).astype(int)
        return df
    except Exception as e:
        print(f"Error in preprocess_yfinance_data: {e}")
        return None
