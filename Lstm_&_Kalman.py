# === Prédiction BTC avec LSTM + Kalman (filtrage + IC) ===

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import cycle
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings("ignore")

# === Chargement des données ===
BTC_data = pd.read_csv(r'Z/BTC/kaggle 1/BTC-USD.csv', skiprows=2)
BTC_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
BTC_data['Date'] = pd.to_datetime(BTC_data['Date'])
BTC_data = BTC_data.set_index('Date')
closedf = BTC_data[['Close']].reset_index()

# === Prétraitement ===
closedf = closedf[closedf['Date'] > '2020-09-13']
close_stock = closedf.copy()
del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

training_size = int(len(closedf) * 0.70)
train_data = closedf[0:training_size, :]
test_data = closedf[training_size:, :1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# === Modèle LSTM ===
model = Sequential()
model.add(LSTM(10, input_shape=(time_step, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=True)

# === Prédiction sur le test ===
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# === Prédiction future (30 jours) ===
x_input = test_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
future_output = []
n_future_days = 30

for i in range(n_future_days):
    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    future_output.append(yhat[0][0])
    temp_input.append(yhat[0][0])

future_output_real = scaler.inverse_transform(np.array(future_output).reshape(-1, 1))
last_date = close_stock['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future_days)

# === Kalman Filter + IC ===
kf = KalmanFilter(initial_state_mean=future_output_real[0, 0],
                  n_dim_obs=1,
                  em_vars=['transition_covariance', 'observation_covariance'])
kf = kf.em(future_output_real, n_iter=5)
filtered_state_means, filtered_state_cov = kf.filter(future_output_real)
conf_int_upper = filtered_state_means.flatten() + 1.96 * np.sqrt(filtered_state_cov[:, 0, 0])
conf_int_lower = filtered_state_means.flatten() - 1.96 * np.sqrt(filtered_state_cov[:, 0, 0])

# === Visualisation ===
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Close (LSTM)': future_output_real.flatten(),
    'Kalman Smoothed': filtered_state_means.flatten(),
    'Upper 95% CI': conf_int_upper,
    'Lower 95% CI': conf_int_lower
})

fig_kalman_ci = px.line(future_df, x='Date', y='Kalman Smoothed',
                        title='🔮 Prédiction BTC (LSTM + Kalman) avec Intervalle de Confiance',
                        labels={'Kalman Smoothed': 'Prix BTC (lissé)'})
fig_kalman_ci.add_scatter(x=future_df['Date'], y=future_df['Upper 95% CI'], mode='lines', name='Borne Supérieure', line=dict(dash='dot'))
fig_kalman_ci.add_scatter(x=future_df['Date'], y=future_df['Lower 95% CI'], mode='lines', name='Borne Inférieure', line=dict(dash='dot'))
fig_kalman_ci.add_scatter(x=future_df['Date'], y=future_df['Predicted Close (LSTM)'], mode='lines', name='LSTM (brut)', line=dict(color='gray', width=1))
fig_kalman_ci.update_layout(plot_bgcolor='white', font_size=14)
fig_kalman_ci.update_xaxes(showgrid=False)
fig_kalman_ci.update_yaxes(showgrid=False)
fig_kalman_ci.show()
