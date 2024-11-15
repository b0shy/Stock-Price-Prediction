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

# Two timestep values (5 and 10)
timesteps_list = [5, 10]
results = {}  # Store performance metrics for each timestep

# Loop over each timestep configuration to build, train, and evaluate models
for timesteps in timesteps_list:
    # Prepare training sequences with the specified timestep
    x_train, y_train = create_sequences(scaled_data, timesteps)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for RNN input

    # Build the RNN model with two layers
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(timesteps, 1)))  # First RNN layer
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(SimpleRNN(units=100))  # Second RNN layer with more units
    model.add(Dropout(0.2))  # Dropout layer
    model.add(Dense(units=1))  # Output layer for predicting the next closing price

    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model with MSE loss

    # Train the model and save the training history for loss visualization
    history = model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1, validation_split=0.2)

    # Prepare test data starting from 2021 for a realistic split, ensuring unseen data for testing
    test_start = dt.datetime(2021, 1, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(company, start=test_start, end=test_end)
    actual_prices = test_data['Close'].values  # Actual closing prices for comparison

    # Combine the entire dataset for testing and transform the 'Close' prices
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - timesteps:].values
    model_inputs = model_inputs.reshape(-1, 1)  # Reshape for scaling
    model_inputs = scaler.transform(model_inputs)  # Scale the combined dataset

    # Create test sequences for prediction based on the `timesteps`
    x_test, y_test = create_sequences(model_inputs, timesteps)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for RNN input

    # Make predictions
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)  # Transform predictions back to original scale
