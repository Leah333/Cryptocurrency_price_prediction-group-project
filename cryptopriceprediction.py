# Import necessary libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
# Must install tensorflow in terminal with pip
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# Must install pmdarima in terminal with pip
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
# Read in the data
df_crypto = pd.read_csv("crypto_dataset.csv", index_col=0)
df_crypto.info()
df_crypto.shape
df_crypto = df_crypto.drop(columns = {'timestamp'})

# Check the first and last five lines of the data
df_crypto.head()
df_crypto.tail()

# Find out which cryptocurrencies have the most samples
crypto_counts = df_crypto['crypto_name'].value_counts()
crypto_counts

# Create a new DataFrame for Bitcoin and visualize the change in prices
df_bitcoin = df_crypto[df_crypto.crypto_name == 'Bitcoin']
df_bitcoin = df_bitcoin.drop(columns={'crypto_name'})
df_bitcoin = df_bitcoin.reset_index(drop=True)
df_bitcoin.info()
df_bitcoin.shape

# Visualize Bitcoin closing prices
plt.figure(figsize=(10,4))
plt.title("Bitcoin Prices")
plt.xlabel("Time")
plt.ylabel("Close")
plt.plot(df_bitcoin["close"])
plt.show()

# Extract the 10 cryptocurrenvies with the most samples
crypto_counts[0:10]

# Plot their closing prices using the log scale for normalization

# Litecoin
df_litecoin = df_crypto[df_crypto.crypto_name == 'Litecoin']
df_litecoin = df_litecoin.drop(columns={'crypto_name'})
df_litecoin = df_litecoin.reset_index(drop=False)

# XRP
df_XRP = df_crypto[df_crypto.crypto_name == 'XRP']
df_XRP = df_XRP.drop(columns={'crypto_name'})
df_XRP = df_XRP.reset_index(drop=False)

# Dogecoin

df_Dogecoin = df_crypto[df_crypto.crypto_name == 'Dogecoin']
df_Dogecoin = df_Dogecoin.drop(columns={'crypto_name'})
df_Dogecoin = df_Dogecoin.reset_index(drop=False)

# Monero
df_monero = df_crypto[df_crypto.crypto_name == 'Monero']
df_monero = df_monero.drop(columns={'crypto_name'})
df_monero = df_monero.reset_index(drop=False)

# Stellar
df_stellar= df_crypto[df_crypto.crypto_name == 'Stellar']
df_stellar = df_stellar.drop(columns={'crypto_name'})
df_stellar = df_stellar.reset_index(drop=True)

# Tether
df_tether = df_crypto[df_crypto.crypto_name == 'Tether']
df_tether = df_tether.drop(columns={'crypto_name'})
df_tether= df_tether.reset_index(drop=True) 

# Ethereum
df_ethereum = df_crypto[df_crypto.crypto_name == 'Ethereum']
df_ethereum = df_ethereum.drop(columns={'crypto_name'})
df_ethereum = df_ethereum.reset_index(drop=True)

# Ethereum Classic
df_ethereum_classic = df_crypto[df_crypto.crypto_name == 'Ethereum Classic']
df_ethereum_classic = df_ethereum_classic.drop(columns={'crypto_name'})
df_ethereum_classic = df_ethereum_classic.reset_index(drop=True)

# Basic Attention Token 
df_BAC = df_crypto[df_crypto.crypto_name == 'Basic Attention Token']
df_BAC = df_BAC.drop(columns={'crypto_name'})
df_BAC = df_BAC.reset_index(drop=True)

plt.plot(df_bitcoin["date"],np.log(df_bitcoin["close"]), label = "Bitcoin")
plt.plot(df_litecoin["date"],np.log(df_litecoin["close"]), label = "Litecoin")
plt.plot(df_XRP["date"],np.log(df_XRP["close"]), label = "XRP")
plt.plot(df_Dogecoin["date"],np.log(df_Dogecoin["close"]), label = "Dogecoin")
plt.plot(df_monero["date"],np.log(df_monero["close"]), label = "Monero")
plt.plot(df_stellar["date"],np.log(df_stellar["close"]), label = "Stellar")
plt.plot(df_tether["date"],np.log(df_tether["close"]), label = "Tether")
plt.plot(df_ethereum["date"],np.log(df_ethereum["close"]), label = "Ethereum")
plt.plot(df_ethereum_classic["date"],np.log(df_ethereum_classic["close"]), label = "Ethereum Classic")
plt.plot(df_BAC["date"],np.log(df_BAC["close"]), label = "Basic Attention Token")

plt.title("Crypto Prices")
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.legend()
plt.show()

# Compare daily returns to determine which cryptocurrency is a better investment
returns_bitcoin = df_bitcoin['close']- df_bitcoin['open']
returns_litecoin = df_litecoin['close']- df_bitcoin['open']
returns_XRP = df_XRP['close']- df_XRP['open']
returns_dogecoin = df_Dogecoin['close'] - df_Dogecoin['open']
returns_monero = df_monero['close'] - df_monero['open']
returns_stellar = df_stellar['close'] - df_stellar['open']
returns_tether = df_tether['close'] - df_tether['open']
returns_ethereum = df_ethereum['close'] - df_ethereum['open']
returns_ethereum_classic = df_ethereum_classic['close']-df_ethereum_classic['open']
returns_BAC = df_BAC['close'] - df_BAC['open']

# Visualize the returns
plt.figure(figsize=(10,4))
plt.title("Returns from Investment")
plt.xlabel("Time")
plt.ylabel("Returns")
returns_bitcoin.plot(label = "Bitcoin")
returns_litecoin.plot(label = "Litecoin")
returns_XRP.plot(label = "XRP")
returns_dogecoin.plot(label = "Dogecoin")
returns_monero.plot(label = "Monero")
returns_stellar.plot(label = "Stellar")
returns_tether.plot(label = "Tether")
returns_ethereum.plot(label = "Ethereum")
returns_ethereum_classic.plot(label = "Ethereum Classic") 
returns_BAC.plot(label = "Basic Attention Token")
plt.legend()
plt.show()

# Visualize volatility
# Compare the percentage increase in the value of the top 3 cryptocurrencies

df_bitcoin['returns'] = (df_bitcoin['close']/df_bitcoin['close'].shift(1)) - 1
df_monero['returns'] = (df_monero['close']/df_monero['close'].shift(1)) - 1
df_ethereum['returns'] = (df_ethereum['close']/df_ethereum['close'].shift(1)) -1
df_bitcoin['returns'].hist(bins = 100, label = 'Bitcoin', alpha = 0.5, figsize = (15,7))
df_monero['returns'].hist(bins = 100, label = 'Monero', alpha = 0.5)
df_ethereum['returns'].hist(bins = 100,label = 'Ethereum',alpha = 0.5)

plt.legend()

# PRICE PREDICTION MODELS

# MODEL USING LINEAR REGRESSION
# Create a new dataframe with the variable to be predicted (closing price)
df_bitcoin2 = df_bitcoin[['close']]

# Variable to predict 30 days into the future
forecast_days = 30

# New column in dataframe for the output, set to the closing price shifted up 30 units
df_bitcoin2['prediction'] = df_bitcoin2.shift(-1*forecast_days)

# Assign the data to arrays
X = np.array(df_bitcoin2.drop(['prediction'],1)) # X consists of values of close
X_scaled = preprocessing.scale(X)                # Input values scaled for normalization
X_pred = X_scaled[-1*forecast_days:]             # Set prediction variable equal to last 30
X_scaled = X_scaled[:-1*forecast_days]           # Remove last 30 from X
y = np.array(df_bitcoin2['prediction'])          # y consists of the predicted output
y = y[:-1*forecast_days]                         # Remove last 30 from y

# Split data into testing and training sets with test_size equal to 20%      
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)

# Train using a Linear Regression model
classifier = LinearRegression()
classifier.fit(X_train, y_train)

# Test for accuracy using r-squared
reg_accuracy = classifier.score(X_test, y_test)
print('accuracy: ', reg_accuracy)

# Predict values
forecast_pred_reg = classifier.predict(X_pred)
print(forecast_pred_reg)

# MODEL USING VECTOR SUPPORT MACHINES
# SVM Model
model = SVR(kernel='rbf', C=1e3, gamma=0.1)
model.fit(X_train, y_train)

# Test for accuracy
svm_accuracy = model.score(X_test, y_test)
print('accuracy:', svm_accuracy)

# Predict values
forecast_pred_svm = model.predict(X_pred)
print(forecast_pred_svm)

# Create new dataframe with predicted values
df_bitcoin_prediction_svm = pd.DataFrame(forecast_pred_reg, columns=['Prediction'])

# Visualize the predictions
X_plot = X[-1*forecast_days:] 
plt.figure(figsize=(10,4))
plt.title('Bitcoin Prices')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.plot(X_plot)
plt.plot(forecast_pred_reg)
plt.plot(forecast_pred_svm)
plt.legend(['True Price', 'LG Prediction', 'SVM Prediction'])

# MODEL USING LSTM
# 70% of the data for training
training_data_len = math.ceil(len(df_bitcoin) * 0.7)
training_data_len

# Scale the data
data = df_bitcoin.filter(['close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

# Create the training dataset
train_data = scaled_data[0:training_data_len, :]

X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60: i, 0])
    y_train.append(train_data[i, 0])


    if i <= 60:
        print(X_train)
        print(y_train)
        print()
        
len(X_train)

X_train, y_train = np.array(X_train), np.array(y_train) # Convert to numpy array
X_train.shape

# LSTM needs 3D, reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape

# Create the testing dataset
test_data = scaled_data[training_data_len - 60 : , :]
X_test = []
y_test = dataset[training_data_len : , :]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60 : i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # Reshape

# Build LSTM model
tf.random.set_seed(123)
bitcoin_model = Sequential()
bitcoin_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
bitcoin_model.add(LSTM(50, return_sequences=False))
bitcoin_model.add(Dense(25))
bitcoin_model.add(Dense(1))

bitcoin_model.compile(optimizer='adam', loss='mse') # Compile the model

# Train the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
history = bitcoin_model.fit(X_train, y_train, batch_size=1, epochs=3)

# Predict price value
bitcoin_predictions = bitcoin_model.predict(X_test)
bitcoin_predictions = scaler.inverse_transform(bitcoin_predictions)
len(bitcoin_predictions)

# Root mean squared error
rmse = np.sqrt(np.mean(bitcoin_predictions - y_test)**2)
rmse

# Plot and visualized the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = bitcoin_predictions
plt.figure(figsize=(10, 4))
plt.title('Bitcoin Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.plot(train)
plt.plot(valid)
plt.legend(['Train', 'Valid', 'Predictions'], loc='upper left')

# Evaluate accuracy
bitcoin_model = tf.keras.metrics.Accuracy()
bitcoin_model.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]] )
bitcoin_model.result().numpy()

# MODEL USING ARIMA

# Set date as index in the bitcoin closing prices dataframe
date1 = df_bitcoin['date']
bitcoin = df_bitcoin[['date','close']]
bitcoin_prices = bitcoin.set_index('date')
bitcoin_prices.sort_index(inplace=True)

# Split the data into training and testing sets
split = int(len(bitcoin_prices)*0.9)
train =bitcoin_prices[0:split]['close']
test = bitcoin_prices[split:]['close']

# Visualize 
plt.plot(bitcoin_prices[0:split]['close'], 'green', label = 'train data')
plt.plot(bitcoin_prices[split:]['close'], 'blue', label = 'test data')
plt.xlabel('date')
plt.ylabel('closing prices')
plt.legend()
plt.show()

# Create the predictive model 
# Check if the data is stationary (Augmented Dickey Fuller Test)
# H0(null hypothesis): time series data is non-stationary

adf = adfuller(train,autolag='AIC')
print(f'ADF Statistic: {adf[0]}')
print(f'n_lags: {adf[1]}')
print(f'p-value: {adf[1]}')
for key, value in adf[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
# Differencing
train_differenced = train.diff().dropna()

# Visualize the differenced data
train_differenced.plot()

# ADF test on differenced data
adf1 = adfuller(train_differenced,autolag='AIC')
print(f'p-value: {adf1[1]}')

# Auto ARIMA
arima_model = auto_arima(train_differenced, start_p = 0, d = 1,start_q = 0, max_p = 5, max_d = 5, max_q = 5,
                        start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, m=12, seasonal=True, error_action = 'warn',
                        trace = True, supress_warnings = True, stepwise = True, random_state = 20, n_fits = 50)

# Prediction
model = ARIMA(train_differenced, order=(5, 1, 0))
predict = pd.DataFrame(model.predict(n_periods = len(test), index = test.index))
predict.columns = ['predicted prices']
predict

# Predicted prices on the test data
plt.plot(train, label = 'Training')
plt.plot(test, label = 'Test')
plt.plot(predict, label = 'Predicted')
plt.legend()
plt.show()
