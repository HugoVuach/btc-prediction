# 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM



import yfinance as yf
import pandas as pd
import requests, json
import numpy as np

import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.subplots import make_subplots

import os
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from itertools import cycle






'''Après avoir appliqué le modèle de régression linéaire à nos données, 
nous allons essayer d'implémenter un modèle de deep learning : 
LSTM qui est très adapté à ce cas.Maintenant nous créons un nouveau 
modèle `Sequential()` puis nous ajoutons 2 couches:
une couche avec l'activation relu avec 10 unités et la deuxieme avec une unité.
'''


# === Téléchargement des données ===
BTC_data = pd.read_csv(r'Z/BTC/kaggle 1/LSTM/BTC-USD.csv', skiprows=2)
BTC_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
BTC_data['Date'] = pd.to_datetime(BTC_data['Date'])
BTC_data = BTC_data.set_index('Date')
closedf = BTC_data[['Close']].reset_index()


##############
# === Stock analysis Chart ===
names_2 = cycle(['BTC Open Price', 'BTC High Price', 'BTC Low Price', 'BTC Close Price'])
fig_2 = px.line(closedf, x=closedf['Date'], y=[closedf['Close'], BTC_data['Open'], BTC_data['High'], BTC_data['Low']],
                labels={'value': 'Price', 'Date': 'Date'})
fig_2.update_layout(title_text='Bitcoin Price Analysis',
                   plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Price')
fig_2.for_each_trace(lambda t: t.update(name=next(names_2)))
fig_2.update_xaxes(showgrid=False)
fig_2.update_yaxes(showgrid=False)

fig_2.show()

##############


print("Forme du dataframe close :", closedf.shape)

# ✅ Filtrer à partir d'une certaine date
closedf = closedf[closedf['Date'] > '2020-09-13']
close_stock = closedf.copy()
del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

print("Shape après normalisation :", closedf.shape)

# ✅ Split train/test
training_size = int(len(closedf) * 0.70)
test_size = len(closedf) - training_size

train_data = closedf[0:training_size, :]
test_data = closedf[training_size:, :1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   #i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

LSTM_model=Sequential()
LSTM_model.add(LSTM(10,input_shape=(None,1),activation="relu"))
LSTM_model.add(Dense(1))
LSTM_model.compile(loss="mean_squared_error",optimizer="adam")
LSTM_model.summary()

history = LSTM_model.fit( X_train, y_train,
                         validation_data=(X_test,y_test),
                         epochs=100, batch_size=32,
                         verbose=True)

predictions = LSTM_model.predict(X_test)
print("\n---------------LSTM---------------\n")
print("Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions)))
print("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions)))+"\n")


# Plotting Loss vs Validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure(figsize=(12, 7))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)

plt.show()

# Faisons la prédiction pour vérifer les mesures de performance.
train_predict=LSTM_model.predict(X_train)
test_predict=LSTM_model.predict(X_test)
train_predict.shape, test_predict.shape

# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

### evaluation 
## Variance Regression Score

print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
print("----------------------------------------------------------------------")

## R square score for regression
print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))
print("----------------------------------------------------------------------")

# Regression loss Mean gamma deviance regression loss MGD
print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")

# Mean Poisson deviance regression loss (MPD)
print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")


# shift train predictions for plotting

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict


# === Prediction plot===
names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': close_stock['Date'],
                       'original_close': close_stock['Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Close price','date': 'Date'})
fig.update_layout(title_text='Prediction avec LSTM',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show() 




# === Prédiction des 30 prochains jours ===


# 🧩 Récupération de la dernière séquence pour démarrer
x_input = test_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])

future_output = []
n_future_days = 5  # nombre de jours à prédire

# 🔁 Boucle de prédiction jour par jour
for i in range(n_future_days):
    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
    yhat = LSTM_model.predict(x_input, verbose=0)
    future_output.append(yhat[0][0])
    temp_input.append(yhat[0][0])  # ajouter la prédiction pour la boucle suivante

# 🔄 Inverser la normalisation pour revenir à l'échelle réelle
future_output_real = scaler.inverse_transform(np.array(future_output).reshape(-1, 1))

# 📅 Générer les dates futures à partir de la dernière date connue
last_date = close_stock['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future_days)

# 📊 Créer un DataFrame pour visualisation
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_output_real.flatten()})

# 📈 Afficher les prédictions avec Plotly
fig_future = px.line(future_df, x='Date', y='Predicted Close',
                     title='📈 Prédiction des 30 prochains jours du prix du BTC',
                     labels={'Predicted Close': 'Prix BTC (USD)'})
fig_future.update_layout(plot_bgcolor='white', font_size=15, font_color='black')
fig_future.update_xaxes(showgrid=False)
fig_future.update_yaxes(showgrid=False)
fig_future.show()

''' 
    un seule feature d'entrée 
    + auto-recursif en mode prediction => la courbe devient lissée,
    + aucun indicateur  evenement ext ( volume, volatilité, indicateur; ect 
    + que 10 neuronne de ma couche 
    => pas à pas que des valeurs tres proches les unes des autres)

    => plan d'action : 
    ✅ Ajoute Open, High, Low, Volume comme entrées, pas juste Close.
    ✅ Utilise des indicateurs techniques.
    ✅ Enrichis ton LSTM : +neurones, +couches.
    ✅ Recalibre MinMaxScaler pour toutes les colonnes.
    ✅ Utilise return_sequences=True + LSTM empilés.


'''

'''
    un modèle LSTM bicouche (128 + 64 unités) produit des résultats raisonnablement précis, avec de bons scores RMSE
    Architectures hybrides (ex : LSTM + CNN + attention) renforcent la capacité à extraire des signaux pertinent
'''