import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, precision_score

def save_plot(fig, filename):
    try:
        os.makedirs('charts', exist_ok=True)
        fig.savefig(f'charts/{filename}', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error in save_plot: {e}")

def evaluate_arima_model(ar_model, steps):
    try:
        forecast = ar_model.get_forecast(steps)
        ypred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        return ypred, conf_int
    except Exception as e:
        print(f"Error in evaluate_arima_model: {e}")
        return None, None

def evaluate_arimax_model(arimax_model, exog, steps):
    try:
        forecast = arimax_model.get_forecast(steps, exog=exog)
        ypred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        return ypred, conf_int
    except Exception as e:
        print(f"Error in evaluate_arimax_model: {e}")
        return None, None

def evaluate_xgb_model(model, test, features):
    try:
        preds = model.predict(test[features])
        preds = pd.Series(preds, index=test.index)
        return preds
    except Exception as e:
        print(f"Error in evaluate_xgb_model: {e}")
        return None

def plot_series(series, label, color='black'):
    fig, ax = plt.subplots()
    ax.plot(series, label=label, color=color)
    ax.legend(loc='upper left')
    return fig

def plot_decomposition(df, trend, seasonal, residual):
    try:
        fig, axs = plt.subplots(4, 1, figsize=(12, 8))
        axs[0].plot(df, label='Original', color='black')
        axs[0].legend(loc='upper left')
        axs[1].plot(trend, label='Trend', color='red')
        axs[1].legend(loc='upper left')
        axs[2].plot(seasonal, label='Seasonal', color='blue')
        axs[2].legend(loc='upper left')
        axs[3].plot(residual, label='Residual', color='black')
        axs[3].legend(loc='upper left')
        return fig
    except Exception as e:
        print(f"Error in plot_decomposition: {e}")
        return None

def plot_forecast(data, forecast, conf_int, label='Forecast'):
    try:
        fig, ax = plt.subplots()
        ax.plot(data, label='Actual')
        ax.plot(forecast, color='orange', label=label)
        ax.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=.15)
        ax.legend(loc='lower right')
        return fig
    except Exception as e:
        print(f"Error in plot_forecast: {e}")
        return None

def plot_xgb_predictions(test, preds):
    try:
        fig, ax = plt.subplots()
        ax.plot(test['Target'], label='Actual')
        ax.plot(preds, label='Predicted')
        ax.legend()
        return fig
    except Exception as e:
        print(f"Error in plot_xgb_predictions: {e}")
        return None
