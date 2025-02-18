import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
import datetime as dt

company = 'AAPL'
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = yf.download(company, start=start, end=end)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediciton_days = 300

x_train = []
y_train = []

# basically this part is taking in 121 data. It will learn from 120 data to predict the 121st data (y_train).

for x in range(prediciton_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediciton_days: x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
#model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(units=50))
#model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing price.

model.compile(optimizer='adam', loss='mean_squared_error')  
model.fit(x_train, y_train, epochs=35, batch_size=32)

test_start = dt.datetime(2021,1,1)      # important
test_end = dt.datetime.now()

test_data = yf.download(company, start=test_start, end=test_end)
actual_price = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediciton_days:].values #last predicition days + len(test_data) data point from the total dataset
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
y_test = []
for x in range(prediciton_days, len(model_inputs)):
  x_test.append(model_inputs[x-prediciton_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

y_test = total_dataset[len(total_dataset) - len(test_data):].values
y_test_norm = scaler.transform(y_test.reshape(-1, 1)) #actual price 

predicted_price_norm = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price_norm) #predicted actual price

mse = mean_squared_error(y_test_norm, predicted_price_norm)
print(f"Mean Squared Error: {mse}")
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

plt.plot(actual_price, color = "black", label=f"Actual {company} Price")
plt.plot(predicted_price, color = "green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

real_data = [model_inputs[len(model_inputs) + 1 - prediciton_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}") #price of company the next day