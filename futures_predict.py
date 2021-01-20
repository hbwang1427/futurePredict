import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from file_load import load_future, load_Daily, load_future_2021
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--market', help='market name (IC, IF, IH).')
parser.add_argument('-d', '--testMarket', help='test market file name.')

args = parser.parse_args()

root_path = 'StockFutureData'
market = args.market
model_name = 'future_price_GRU'
if market:
    model_name = model_name + '_'+market
model_name_h5= model_name+'.h5'

#load tick data
mypath = os.path.join(root_path, market)
prices = load_future(mypath)
prices = prices[['最新','持仓','增仓','成交额','成交量']]

# preparing label data
prices_shift = prices.shift(-1)
label = prices_shift['最新']

# adjusting the shape of both
prices.drop(prices.index[len(prices)-1], axis=0, inplace=True)
label.drop(label.index[len(label)-1], axis=0, inplace=True)

# conversion to numpy array
x, y = prices.values, label.values

# scaling values for model
x_scale = MinMaxScaler()
y_scale = MinMaxScaler()

X_train = x_scale.fit_transform(x)
y_train = y_scale.fit_transform(y.reshape(-1,1))

# load test
test_prices = load_future_2021(args.testMarket)
openInterest_shift = test_prices['openInterest'].shift(-1)
test_prices = [test_prices[['lastPrice']], test_prices[['openInterest']], test_prices[['openInterest']]-openInterest_shift, test_prices[['turnOver']], test_prices[['totalVol']]].toArray()
test_prices_shift = test_prices.shift(-1)
test_label = test_prices_shift['lastPrice']
test_prices.drop(test_prices.index[len(test_prices)-1], axis=0, inplace=True)
test_label.drop(test_label.index[len(test_label)-1], axis=0, inplace=True)
# conversion to numpy array
xtest, ytest = test_prices.values, test_label.values
X_test = x_scale.fit_transform(xtest)
y_test = y_scale.fit_transform(ytest.reshape(-1,1))

X_train = X_train.reshape((-1,1,5))
X_test = X_test.reshape((-1,1,5))

model = load_model(model_name_h5)

score = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
y_test = y_scale.inverse_transform(y_test)
plt.plot(yhat[-100:], label='Predicted')
plt.plot(y_test[-100:], label='Ground Truth')
plt.legend()
plt.show()