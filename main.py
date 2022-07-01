# Author Brianna Stegemann
# WGU C964 Capstone Project

# Install the necessary libraries to run the project
import datetime
import os
import datetime as dt

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from plotly import graph_objs as go
import pandas as pd
from pandas_datareader import data
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import streamlit as st
from keras.models import load_model
import math
import cufflinks as cf

# Set the page configuration.
app_name = 'Stock Prediction Application'
st.set_page_config(page_title=app_name, layout='wide', initial_sidebar_state='expanded')
st.sidebar.title(app_name)

# USER INPUT
# Create a list of Stock Tickers to choose from.
Tickers = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'TSLA']

# Select Ticker
ticker = st.sidebar.selectbox('Select Ticker', sorted(Tickers), index=0)

# This method will import data from yahoo finance and add/remove necessary columns.
# This is our descriptive method as it parses/adds/changes data.

# Define start and end points, this will be Apple's stock for the last 4 years
start_date = st.sidebar.date_input('Start date', datetime.datetime(2019, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())

# Create the dataframe.
df = data.DataReader(ticker, 'yahoo', start_date, end_date)  # Define the stock of Apple for now to create the model

st.header(f'{ticker} Stock Price')

# Inspect the raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

# df['RSI'] = rsi(df)

# Interactive data visualizations using cufflinks.
# Visualization #1 : Candlestick chart of the data.
qf = cf.QuantFig(df, legend='top', name=ticker)

# Add Relative Strength Indicator (RSI) study.
qf.add_rsi(periods=14, color='magenta')

# Add Bollinger Bands (BOLL) to the chart.
qf.add_bollinger_bands(periods=20, boll_std=2, colors=['java', 'grey'], fill=True)

# Add 'volume' study to the chart.
qf.add_volume()

fig = qf.iplot(asFigure=True, dimensions=(800, 600))

# Render plot
st.plotly_chart(fig)

# Visualization # 2: Line graph of Closing Price with 100 and 200 days Moving Average.
st.subheader('Closing Price vs Time Chart with 50MA & 200MA')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'b', label='Closing price of' + ticker)
ma50 = df.Close.rolling(50).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(ma50, 'magenta', label='50 Moving Average')
plt.plot(ma200, 'grey', label='200 Moving Average')
plt.legend()
st.pyplot(fig2)

# Splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0: int(len(df) * 0.70)])  # 70% training data
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])  # 30 % of the training data

# Scale down the data
from sklearn.preprocessing import MinMaxScaler

# Define an object
scaler = MinMaxScaler(feature_range=(0, 1))

# Data Training Array
data_training_array = scaler.fit_transform(data_training)

# Load my ML model
model = load_model('sp_keras_model.h5')

# Testing the ML model part.
# Getting stock information for the past 100 days.
past_100_days = data_training.tail(100)

# Append the testing data
final_df = past_100_days.append(data_testing, ignore_index=True)

# Apply MinMaxScaler
input_data = scaler.fit_transform(final_df)

# Define the testing data
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

# Convert to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions

y_predicted = model.predict(x_test)

# Get the factor
scaler = scaler.scale_

# Divide y_test and y_predicted by the factor
scale_factor = 1 / scaler[0]

# Multiply y_predicted and y_test by scale_factor.
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Get the root mean squared error
st.caption('Root Mean Squared Error of the LSTM Model.')
rmse = np.sqrt(np.mean(((y_predicted - y_test) ** 2)))
rmse

# Get mape (Mean Absolute Percentage Error)
st.caption('Mean Absolute Percentage Error of the LSTM Model.')
mape = np.mean(np.abs((y_test - y_predicted) / y_test)) * 100
mape

# Third Visualization: Line Graph
st.subheader('Predictions vs Original')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel(ticker + 'Price')
plt.legend()
st.pyplot(fig3)
