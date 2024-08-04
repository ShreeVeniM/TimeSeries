import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.data_loader import load_data, load_yfinance_data
from src.preprocess import preprocess_data, preprocess_yfinance_data
from src.train import train_arima_model, train_arimax_model, train_xgb_model
from src.evaluate import evaluate_arima_model, evaluate_arimax_model, evaluate_xgb_model, plot_series, plot_decomposition, plot_forecast, plot_xgb_predictions, save_plot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load and preprocess data
        logging.info("Loading data from AAPL.csv")
        df = load_data('src/dataset/AAPL.csv')
        if df is None:
            return
        df = preprocess_data(df)
        if df is None:
            return
        logging.info("Data loaded and preprocessed successfully")

        # Univariate analysis
        logging.info("Performing univariate analysis")
        fig = plot_series(df['AAPL'], label='AAPL')
        save_plot(fig, 'univariate_analysis.png')
        logging.info("Univariate analysis plot saved")

        # Decomposition
        logging.info("Performing seasonal decomposition")
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposed = seasonal_decompose(df['AAPL'])
        fig = plot_decomposition(df['AAPL'], decomposed.trend, decomposed.seasonal, decomposed.resid)
        save_plot(fig, 'decomposition.png')
        logging.info("Seasonal decomposition plots saved")

        # ACF and PACF plots
        logging.info("Plotting ACF and PACF")
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(df['AAPL'].dropna(), ax=ax[0])
        plot_pacf(df['AAPL'].dropna(), lags=11, ax=ax[1])
        save_plot(fig, 'acf_pacf.png')
        logging.info("ACF and PACF plots saved")

        # Augmented Dickey-Fuller (ADF) test
        logging.info("Performing Augmented Dickey-Fuller (ADF) test")
        from statsmodels.tsa.stattools import adfuller
        results = adfuller(df['AAPL'])
        logging.info(f'ADF p-value: {results[1]}')

        # 1st order differencing
        logging.info("Performing 1st order differencing")
        v1 = df['AAPL'].diff().dropna()
        results1 = adfuller(v1)
        logging.info(f'ADF p-value after differencing: {results1[1]}')

        # Plot the differenced series
        logging.info("Plotting the differenced series")
        fig = plot_series(v1, label='1st Order Differenced Series')
        save_plot(fig, 'differenced_series.png')
        logging.info("Differenced series plot saved")

        # ARIMA Model
        logging.info("Training ARIMA model")
        ar_model = train_arima_model(df['AAPL'], order=(1, 1, 1))
        if ar_model is None:
            return
        logging.info(f"ARIMA model summary:\n{ar_model.summary()}")
        ypred, conf_int = evaluate_arima_model(ar_model, steps=2)
        dp = pd.DataFrame({
            'price_actual': [184.40, 185.04],
            'price_predicted': ypred.values,
            'lower_int': conf_int['lower AAPL'].values,
            'upper_int': conf_int['upper AAPL'].values
        }, index=pd.to_datetime(['2024-01-01', '2024-02-01']))
        logging.info(f"ARIMA model predictions:\n{dp}")

        # Plot forecast
        logging.info("Plotting ARIMA model forecast")
        fig = plot_forecast(df['AAPL'], dp['price_predicted'], dp[['lower_int', 'upper_int']])
        save_plot(fig, 'arima_forecast.png')
        logging.info("ARIMA forecast plot saved")

        # Evaluate ARIMA model
        logging.info("Evaluating ARIMA model")
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(dp['price_actual'], dp['price_predicted'])
        logging.info(f'ARIMA MAE = {mae}')

        # Load yfinance data
        logging.info("Loading data from Yahoo Finance")
        yf_data = load_yfinance_data("AAPL", start="2000-01-01", end="2022-05-31")
        if yf_data is None:
            return
        yf_data = preprocess_yfinance_data(yf_data)
        if yf_data is None:
            return
        logging.info("Yahoo Finance data loaded and preprocessed successfully")

        # Split data into train and test sets
        logging.info("Splitting data into train and test sets")
        train = yf_data.iloc[:-30]
        test = yf_data.iloc[-30:]
        features = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Train XGBoost model
        logging.info("Training XGBoost model")
        xgb_model = train_xgb_model(train, features)
        if xgb_model is None:
            return

        # Evaluate XGBoost model
        logging.info("Evaluating XGBoost model")
        preds = evaluate_xgb_model(xgb_model, test, features)
        if preds is None:
            return

        # Plot XGBoost predictions
        logging.info("Plotting XGBoost model predictions")
        fig = plot_xgb_predictions(test, preds)
        save_plot(fig, 'xgb_predictions.png')
        logging.info("XGBoost predictions plot saved")

        # Evaluate XGBoost model
        logging.info("Calculating precision for XGBoost model")
        from sklearn.metrics import precision_score
        precision = precision_score(test['Target'], preds)
        logging.info(f'XGBoost Precision = {precision}')

    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
