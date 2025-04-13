import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import plotly.express as px

keras.utils.set_random_seed(812)

# Load the dataset
data = pd.read_csv("/users/vaibhavmohankumar/desktop/US-Inflation-LSTM/US CPI.csv")

#Removes any null values in the dataset
data.isnull().sum()

# Convert the 'date' column to datetime format
data['Yearmon'] = pd.to_datetime(data['Yearmon'])

# Extract year and inflation rate
data['year'] = data['Yearmon'].dt.year

data = data[['year', 'CPI']]

# Set the 'year' column as the index
data.set_index('year', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# Split data into train and test sets
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]  #[a:b] a is included, b is not included

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12  # Length of sequences for LSTM
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)


# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=50))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=50, batch_size=32,validation_data=(X_test,y_test))


# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Year': np.concatenate((data.index[seq_length:seq_length+len(train_predict)], data.index[seq_length+len(train_predict):seq_length+len(train_predict)+len(test_predict)])),
    'Predicted CPI': np.concatenate((train_predict.flatten(), test_predict.flatten())),
    'Actual CPI': np.concatenate((y_train.flatten(), y_test.flatten()))
})
print(plot_data)

# Create an interactive line plot using Plotly Express
fig = px.line(plot_data, x='Year', y=['Predicted CPI', 'Actual CPI'], title=' US CPI Forecasting with Long Short-Term Memory Model (LSTM)')
fig.update_layout(xaxis_title='Year', yaxis_title=' US CPI', legend_title='Data')
fig.show()


# Create an interactive plot for training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()