import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler

# Define the time range for data retrieval
start_date = dt.datetime(2017, 1, 1)
end_date = dt.datetime(2024, 1, 1)

# Setting up the title for Streamlit app
st.title('Dynamic Stock Price Predictor')

# User input for stock ticker
stock_ticker = st.text_input('Enter Stock Ticker (e.g., TSLA, BTC-USD, ETH-USD):', 'TSLA')

# Validating the user input
if stock_ticker.lower() == 'yahoo':
  st.warning("Invalid ticker. Please enter a different stock ticker.")
else:
  # Fetching data using yfinance
  stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

  # Displaying the stock data and its summary
  st.subheader('Data Overview (2017-2024)')
  st.write(stock_data.describe())
  st.write(stock_data.tail(20))

  # Plotting the Closing Price trend
  st.subheader('Closing Price Trend')
  close_price_fig = plt.figure(figsize=(12, 6))
  plt.plot(stock_data['Close'], label='Close Price')
  plt.legend()
  st.pyplot(close_price_fig)

  # Preparing data for model prediction
  close_data = stock_data.filter(['Close'])
  close_dataset = close_data.values
  training_length = math.ceil(len(close_dataset) * 0.8)

  # Scaling the data
  data_scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_close_data = data_scaler.fit_transform(close_dataset)

  # Loading the pre-trained model
  stock_model = load_model('model_bitcoin.h5')

  # Preparing the test dataset
  test_close_data = scaled_close_data[training_length - 60:, :]
  test_features = []
  actual_prices = close_dataset[training_length:, :]
  for i in range(60, len(test_close_data)):
    test_features.append(test_close_data[i-60:i, 0])

  test_features = np.array(test_features)
  test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

  # Making future price predictions
  last_60_days_data = close_data[-60:].values
  last_60_days_scaled = data_scaler.fit_transform(last_60_days_data)
  future_test_data = []
  future_test_data.append(last_60_days_scaled)
  future_test_data = np.array(future_test_data)
  future_test_data = np.reshape(future_test_data, (future_test_data.shape[0], future_test_data.shape[1], 1))
  future_price_prediction = stock_model.predict(future_test_data)
  future_price_prediction = data_scaler.inverse_transform(future_price_prediction)

  # Displaying the prediction and original prices
  st.subheader('Original vs Predicted Price')
  comparison_fig = plt.figure(figsize=(12, 6))
  plt.plot(actual_prices, 'blue', label='Original Price')
  plt.plot(future_price_prediction, 'red', label='Predicted Price')
  plt.xlabel('Time Period')
  plt.ylabel('Price')
  plt.legend()
  st.pyplot(comparison_fig)
