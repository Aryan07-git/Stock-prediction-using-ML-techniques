import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from arima_model import train_arima_model, forecast_arima_model
from xgboost_model import train_xgboost_model, forecast_xgboost_model

st.set_page_config(layout="wide")
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://wallpapers.com/images/featured/dark-night-80kcxoa2szb17vmq.jpg");
    background-size: 100vw 100vh;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data


def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler


def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        tf.keras.layers.Dropout(0.2),  # Dropout layer to prevent overfitting
        Dense(50, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model



def plot_forecast(data, forecast, n_months, scaler, title):
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=n_months * 30)
    forecast_data = scaler.inverse_transform(forecast.reshape(-1, 1))
    fig = go.Figure([
        go.Scatter(x=forecast_dates, y=forecast_data.flatten(), name='Forecast', mode='lines')
    ])
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig


def plot_raw_data(data, title=None):
    fig = go.Figure([
        go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open Price', line=dict(color='yellow')),
        go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='red'))
    ])
    if title:
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig


def warning_message():
    st.sidebar.warning("⚠️ Note:")
    st.sidebar.markdown("""
    This is to inform you that accurate prediction of stock market returns is very challenging due to the volatile and non-linear nature of financial stock markets. 
    While our app uses RNN-based algorithms and takes factors like inflation into account to provide forecasts, the outcomes should not be relied upon for real-life trading decisions.
    """)


st.markdown("""
    <style>
    .flex-container {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .flex-container img {
        margin-right: 20px;
    }
    </style>
    <div class="flex-container">
        <img src="https://media.istockphoto.com/id/1363959942/vector/genie-granting-three-wishes.jpg?s=612x612&w=0&k=20&c=tJnzxQSPeAypPx6rP41ip3NV-tLGX9X7Bq0ZjHRnrTE=" alt="Genie" width="200">
        <h1 style='font-size: 60px; color: yellow;'>FORECAST GENIE</h1>
    </div>
    """, unsafe_allow_html=True)

st.write("""
Welcome to the Stock Forecast App!

This project is aimed at providing users with predictions and insights into the stock market by using various models including LSTM, ARIMA, and XGBoost. Our app enables you to analyze historical stock data, visualize trends, and make informed decisions about your investments. Whether you're a seasoned investor or just starting out, this app offers valuable tools to help you navigate the complex world of stock trading.

Our Web App is capable of offering:
- Select Dataset: Choose a stock dataset from the dropdown menu.
- Months of Prediction: Use the dropdown to select the number of months you want to forecast.
- Raw Data: Displayed below includes the selected stock's data and raw data plot.

Feel free to explore and make the most out of our Stock Forecast App!
""")

warning_message()


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))



selected_stock = st.selectbox('Select dataset for prediction', ['None', 'GME', 'META', 'HAL', 'MSFT', 'AAPL', 'GOOG'])

if selected_stock != 'None':
    n_months = st.selectbox('Months of prediction:', [1, 2, 3, 4, 5, 6], index=0)
    data = load_data(selected_stock)
    st.markdown(f"<h2 style='color: red; font-size: 40px;'>{selected_stock} Data</h2>", unsafe_allow_html=True)
    st.write(data.tail())
    st.markdown("<h2 style='color: red; font-size: 40px;'>Raw Data Plot</h2>", unsafe_allow_html=True)
    st.plotly_chart(plot_raw_data(data, f'{selected_stock} Raw Data'))

    if st.button(f"Generate Forecast for {selected_stock}"):
        st.write("Opting for options with longer time horizons may result in less accurate predictions")
        gif_file = "monkey.gif"
        gif_placeholder = st.image(gif_file, width=400)
        df_train = data[['Date', 'Close']]
        scaled_data, scaler = preprocess_data(df_train)
        time_step = 30

        X_train, y_train = [], []
        for i in range(len(scaled_data) - time_step):
            X_train.append(scaled_data[i:(i + time_step), 0])
            y_train.append(scaled_data[i + time_step, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model_lstm = create_lstm_model((X_train.shape[1], 1))
        model_lstm.fit(X_train, y_train, epochs=20, batch_size=64,verbose=2)

        lstm_forecast_input = scaled_data[-time_step:].reshape((1, time_step, 1))
        lstm_forecast = []
        for _ in range(n_months * 30):
            next_day_lstm = model_lstm.predict(lstm_forecast_input)
            lstm_forecast.append(next_day_lstm[0])
            lstm_forecast_input = np.append(lstm_forecast_input[:, 1:, :], [[next_day_lstm[0]]], axis=1)

        lstm_forecast = np.array(lstm_forecast)

        st.subheader(f'{selected_stock} LSTM Forecast Plot', divider='rainbow')
        st.plotly_chart(plot_forecast(data, lstm_forecast, n_months, scaler, f'{selected_stock} LSTM Forecast'))
        gif_placeholder.empty()

        y_true = data['Close'].values[-len(lstm_forecast):]
        y_pred = scaler.inverse_transform(lstm_forecast)
        mape = calculate_mape(y_true, y_pred)
        st.write(f"MAPE for {selected_stock} LSTM Forecast: {mape:.2f}%")

st.sidebar.subheader('Comparison Options',divider='rainbow')
st.markdown("""
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: brown;
}
</style>
""", unsafe_allow_html=True)

if st.sidebar.selectbox('Select Comparison Type',
                        ['None', 'Compare with Another Stock']) == 'Compare with Another Stock':
    selected_comparison_stock = st.sidebar.selectbox('Select dataset for comparison',
                                                     ['GOOG', 'AAPL', 'MSFT', 'GME', 'HAL', 'META'])
    selected_data = load_data(selected_stock)
    comparison_data = load_data(selected_comparison_stock)

    st.markdown(
        f"<h2 style='color: yellow; font-size: 20px;'>Comparing {selected_stock} with {selected_comparison_stock}</h2>",
        unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f'{selected_stock} Data',divider='rainbow')
        st.write(selected_data.tail())
        st.subheader('Raw Data Plot',divider='rainbow')
        st.plotly_chart(plot_raw_data(selected_data, f'{selected_stock} Raw Data'))

        if st.button(f"Generate Forecast for {selected_stock} in Comparison"):
            st.write("Opting for options with longer time horizons may result in less accurate predictions")
            gif_file = "mon2.gif"
            gif_placeholder = st.image(gif_file, width=400)
            df_train_selected = selected_data[['Date', 'Close']]
            scaled_data_selected, scaler_selected = preprocess_data(df_train_selected)
            X_train_selected, y_train_selected = [], []
            for i in range(len(scaled_data_selected) - 30):
                X_train_selected.append(scaled_data_selected[i:(i + 30), 0])
                y_train_selected.append(scaled_data_selected[i + 30, 0])
            X_train_selected, y_train_selected = np.array(X_train_selected), np.array(y_train_selected)
            X_train_selected = np.reshape(X_train_selected, (X_train_selected.shape[0], X_train_selected.shape[1], 1))

            model_lstm_selected = create_lstm_model((X_train_selected.shape[1], 1))
            model_lstm_selected.fit(X_train_selected, y_train_selected, epochs=100, batch_size=64)

            lstm_forecast_input_selected = scaled_data_selected[-30:].reshape((1, 30, 1))
            lstm_forecast_selected = []
            for _ in range(n_months * 30):
                next_day_lstm_selected = model_lstm_selected.predict(lstm_forecast_input_selected)
                lstm_forecast_selected.append(next_day_lstm_selected[0])
                lstm_forecast_input_selected = np.append(lstm_forecast_input_selected[:, 1:, :],
                                                         [[next_day_lstm_selected[0]]], axis=1)

            lstm_forecast_selected = np.array(lstm_forecast_selected)

            st.subheader(f'{selected_stock} LSTM Forecast Plot in Comparison',divider='rainbow')
            st.plotly_chart(plot_forecast(selected_data, lstm_forecast_selected, n_months, scaler_selected,
                                          f'{selected_stock} LSTM Forecast in Comparison'))
            gif_placeholder.empty()

            y_true_selected = selected_data['Close'].values[-len(lstm_forecast_selected):]
            y_pred_selected = scaler_selected.inverse_transform(lstm_forecast_selected)
            mape_selected = calculate_mape(y_true_selected, y_pred_selected)
            st.write(f"MAPE for {selected_stock} LSTM Forecast in Comparison: {mape_selected:.2f}%")

    with col2:
        st.subheader(f'{selected_comparison_stock} Data',divider='rainbow')
        st.write(comparison_data.tail())
        st.subheader('Raw Data Plot',divider='rainbow')
        st.plotly_chart(plot_raw_data(comparison_data, f'{selected_comparison_stock} Raw Data'))

        if st.button(f"Generate Forecast for {selected_comparison_stock} in Comparison"):
            st.write("Opting for options with longer time horizons may result in less accurate predictions")
            gif_file = "mon2.gif"
            gif_placeholder = st.image(gif_file, width=400)
            df_train_comparison = comparison_data[['Date', 'Close']]
            scaled_data_comparison, scaler_comparison = preprocess_data(df_train_comparison)
            X_train_comparison, y_train_comparison = [], []
            for i in range(len(scaled_data_comparison) - 30):
                X_train_comparison.append(scaled_data_comparison[i:(i + 30), 0])
                y_train_comparison.append(scaled_data_comparison[i + 30, 0])
            X_train_comparison, y_train_comparison = np.array(X_train_comparison), np.array(y_train_comparison)
            X_train_comparison = np.reshape(X_train_comparison,
                                            (X_train_comparison.shape[0], X_train_comparison.shape[1], 1))

            model_lstm_comparison = create_lstm_model((X_train_comparison.shape[1], 1))
            model_lstm_comparison.fit(X_train_comparison, y_train_comparison, epochs=100, batch_size=64)

            lstm_forecast_input_comparison = scaled_data_comparison[-30:].reshape((1, 30, 1))
            lstm_forecast_comparison = []
            for _ in range(n_months * 30):
                next_day_lstm_comparison = model_lstm_comparison.predict(lstm_forecast_input_comparison)
                lstm_forecast_comparison.append(next_day_lstm_comparison[0])
                lstm_forecast_input_comparison = np.append(lstm_forecast_input_comparison[:, 1:, :],
                                                           [[next_day_lstm_comparison[0]]], axis=1)

            lstm_forecast_comparison = np.array(lstm_forecast_comparison)

            st.subheader(f'{selected_comparison_stock} LSTM Forecast Plot in Comparison',divider='rainbow')
            st.plotly_chart(plot_forecast(comparison_data, lstm_forecast_comparison, n_months, scaler_comparison,
                                          f'{selected_comparison_stock} LSTM Forecast in Comparison'))
            gif_placeholder.empty()

            y_true_comparison = comparison_data['Close'].values[-len(lstm_forecast_comparison):]
            y_pred_comparison = scaler_comparison.inverse_transform(lstm_forecast_comparison)
            mape_comparison = calculate_mape(y_true_comparison, y_pred_comparison)
            st.write(f"MAPE for {selected_comparison_stock} LSTM Forecast in Comparison: {mape_comparison:.2f}%")

st.sidebar.subheader('ARIMA & XGBoost Prediction',divider='rainbow')
selected_prediction_type = st.sidebar.selectbox('Select prediction type', ['None', 'ARIMA', 'XGBoost'])

if selected_prediction_type != 'None' and selected_stock != 'None':
    n_months = st.sidebar.selectbox('Months of prediction for ARIMA & XGBoost:', [1, 2, 3, 4, 5, 6], index=0)
    data = load_data(selected_stock)
    scaled_data, scaler = preprocess_data(data[['Date', 'Close']])
    time_step = 30

    if selected_prediction_type == 'ARIMA':
        arima_forecast = forecast_arima_model(data['Close'], n_months)
        st.subheader(f'{selected_stock} ARIMA Forecast Plot',divider='rainbow')
        st.plotly_chart(plot_forecast(data, arima_forecast, n_months, scaler, f'{selected_stock} ARIMA Forecast'))
        st.write("(When the historical data doesn't have strong seasonality the ARIMA forecasting model may find it difficult to predict the future.)")

        y_true = data['Close'].values[-len(arima_forecast):]
        y_pred = scaler.inverse_transform(arima_forecast.reshape(-1, 1))

    elif selected_prediction_type == 'XGBoost':
        xgboost_forecast = forecast_xgboost_model(np.reshape(data[['Close']].values, (-1, 1)), n_months)
        st.subheader(f'{selected_stock} XGBoost Forecast Plot',divider='rainbow')
        st.plotly_chart(plot_forecast(data, xgboost_forecast, n_months, scaler, f'{selected_stock} XGBoost Forecast'))
        y_true = data['Close'].values[-len(xgboost_forecast):]
        y_pred = scaler.inverse_transform(xgboost_forecast)

st.markdown(""" --- """)
st.write("""
    This Stock Forecast App is developed by Aryan Negi and Rishabh. 
    We aim to provide an easy-to-use tool for stock market analysis and prediction using state-of-the-art machine learning models.
    """)
