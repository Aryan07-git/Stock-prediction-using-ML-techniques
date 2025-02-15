import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_xgboost_model(X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    return model
def forecast_xgboost_model(data, n_months):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_reshaped = data.reshape(-1, 1)
    data_scaled = scaler.fit_transform(data_reshaped)

    n_steps = 30
    X, y = [], []
    for i in range(len(data_scaled) - n_steps):
        X.append(data_scaled[i:i + n_steps, 0])
        y.append(data_scaled[i + n_steps, 0])
    X, y = np.array(X), np.array(y)

    model = train_xgboost_model(X, y)
    forecast = []
    current_batch = data_scaled[-n_steps:]
    for _ in range(n_months * 30):
        current_batch_reshaped = current_batch.reshape((1, n_steps))
        pred = model.predict(current_batch_reshaped)[0]
        forecast.append(pred)
        current_batch = np.append(current_batch[1:], pred).reshape(-1, 1)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

