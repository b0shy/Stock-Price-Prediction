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

