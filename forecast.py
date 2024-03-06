# forecast.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load your financial time series dataset
# Modify this line to load your specific dataset
data = pd.read_csv('data/financial_data.csv')

# Extract relevant features (adjust as needed based on your dataset)
# For simplicity, let's assume 'Close' prices as the target variable
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Function to create LSTM dataset with time steps
def create_lstm_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

# Define the number of time steps and split the dataset
time_steps = 10  # Adjust as needed
X, y = create_lstm_dataset(prices_scaled, time_steps)

# Reshape the data for LSTM input (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Make predictions on the entire dataset
predicted_prices_scaled = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

# Evaluate the model performance
mae = mean_absolute_error(prices[time_steps:], predicted_prices)
rmse = np.sqrt(mean_squared_error(prices[time_steps:], predicted_prices))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Visualize the results
plt.plot(data['Date'][time_steps:], prices[time_steps:], label='Actual Prices')
plt.plot(data['Date'][time_steps:], predicted_prices, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Financial Forecasting with LSTM')
plt.legend()
plt.show()
