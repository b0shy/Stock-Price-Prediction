import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load historical stock price data from Yahoo Finance
company = 'META'  # Specify the stock ticker symbol
start = dt.datetime(2012, 1, 1)
end = dt.datetime.now()

# Download the stock data within the specified date range
data = yf.download(company, start=start, end=end)

# Scale the 'Close' price data to values between 0 and 1 for normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Function to create sequences for time series forecasting based on timesteps
def create_sequences(data, timesteps):
    x, y = [], []
    for i in range(timesteps, len(data)):
        x.append(data[i - timesteps:i, 0])  # Create sequences of length `timesteps`
        y.append(data[i, 0])  # Target value is the price after the sequence
    return np.array(x), np.array(y)
