import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

def fetch_stock_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)
    return data
ticker = 'MSFT'
start_date = '2010-01-01'
st.title('📈Stock Forecasting using LSTMs')
user_input = st.text_input('Enter Stock Ticker', 'MSFT')
data = fetch_stock_data(user_input, start_date)

st.subheader('Data since 2010')
st.write(data.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (10,8))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100MA chart')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(10,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time with 200MA chart')
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(10,6))
plt.plot(data.Close)
plt.plot(ma200, 'g')
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100MA & 200MA chart')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(10,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

data_train = pd.DataFrame(data = data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data = data['Close'][int(len(data)*0.80):int(len(data))])
print(data_train.shape)
print(data_test.shape)

scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

model = load_model('keras_model.h5')

past_100_days = data_train.tail(100)
final_data = pd.concat([past_100_days, data_test], ignore_index=True)
input = scaler.fit_transform(final_data)

x_test = []
y_test = []

for i in range(100, input.shape[0]):
    x_test.append(input[i-100:i])
    y_test.append(input[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_prediction = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_prediction = y_prediction * scale_factor
y_test = y_test * scale_factor

st.subheader('Original vs Predictions')
fig2 = plt.figure(figsize=(10,6))
plt.plot(y_test, 'g', label = 'Actual Price')
plt.plot(y_prediction, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)