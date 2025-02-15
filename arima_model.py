
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(data, order=(1, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima_model(data, n_months):
    model_fit = train_arima_model(data)
    forecast_steps = n_months * 30
    forecast = model_fit.forecast(steps=forecast_steps)
    return forecast.values
