import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

def train_arima_model(series, order):
    try:
        model = ARIMA(series, order=order)
        ar_model = model.fit()
        return ar_model
    except Exception as e:
        print(f"Error in train_arima_model: {e}")
        return None

def train_arimax_model(endog, exog, order):
    try:
        model = ARIMA(endog, exog=exog, order=order)
        arimax_model = model.fit()
        return arimax_model
    except Exception as e:
        print(f"Error in train_arimax_model: {e}")
        return None

def train_xgb_model(train, features):
    try:
        model = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
        model.fit(train[features], train['Target'])
        return model
    except Exception as e:
        print(f"Error in train_xgb_model: {e}")
        return None
