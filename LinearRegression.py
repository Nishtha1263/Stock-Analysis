import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ticker = "GOOG" # Change this to the ticker symbol of the company you want to predict
start_date = "2020-01-01"
end_date = "2023-01-01"

data = yf.download(ticker, start=start_date, end=end_date)
print(data)

# Prepare data for the model
data.dropna(inplace = True)
data['Days'] = (data.index - pd.to_datetime(start_date)).days
X = data['Days'].values.reshape(-1, 1)
y = data['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

# Predict the stock price for a specific day
future_day = 1000
predicted_price = regressor.predict([[future_day]])
print(f"Predicted stock price for day {future_day}: ${predicted_price[0]:.2f}")

from sklearn.metrics import r2_score

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared score:", r2)

import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='blue', label='Actual Stock Price')
plt.scatter(X_test, y_pred, color='red', label='Predicted Stock Price')
plt.xlabel('Days since ' + start_date)
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction (' + ticker + ')')
plt.legend()
plt.show()