# -*- coding: utf-8 -*-
"""BITCOIN PRICE PREDICTION MODEL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CeawJhBZPtMUc0enmG5P_UQ6KnRBxpyQ
"""

pip install yfinance

import yfinance as yf
df = yf.download('BTC-USD')

df.head()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.plot(df.Close)



"""Train Test Split"""

to_row=int(len(df)*0.9)
train = df[0:to_row]['Adj Close']
test = df[to_row:]['Adj Close']
training = list(df[0:to_row]['Adj Close'])
testing = list(df[to_row:]['Adj Close'])

"""Proper Visualisation"""

plt.plot(train,color = 'red', label = 'Train Data')
plt.plot(test,color = 'green', label = 'Test Data')
plt.xlabel('Dates')
plt.ylabel('closing prices')
plt.legend()
plt.grid(True)
plt.figure(figsize=(10,6))

model_predictions = []
n_test_observ = len(test)
import math

for i in range(n_test_observ):
  model = ARIMA(training,order = (4,1,0))
  model_fit = model.fit()
  output = model_fit.forecast()
  yhat = list(output[0])[0]
  model_predictions.append(yhat)
  actual_test_value = testing[i]
  training.append(actual_test_value)

print(model_fit.summary())

"""VISUALISATION OF THE Prediction"""

plt.figure(figsize=(15,9))
plt.grid(True)

data_range = df[to_row:].index

plt.plot(data_range,model_predictions[:302],color = 'blue',marker = 'o',linestyle = 'dashed',label = 'BTC PREDICTED PRICE')
plt.plot(data_range,testing,color = 'red', label = 'BTC ANNUAL PRICE')
plt.title('BTC PRICE PREDICTION')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
plt.show

len(testing)

len(model_predictions)

