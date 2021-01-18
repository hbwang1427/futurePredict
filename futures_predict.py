import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from file_load import load_future, load_Daily
import os

root_path = 'StockFutureData'
market = 'IC'
dailyMarket = ''
model_name = 'future_price_GRU'+'_'+market+dailyMarket
model_name_h5= model_name+'.h5'

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

X = x_scale.fit_transform(x)
Y = y_scale.fit_transform(y.reshape(-1,1))

# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01)
X_train = X_train.reshape((-1,1,5))
X_test = X_test.reshape((-1,1,5))

if not os.path.exists(model_name_h5):
    # creating model using Keras
    # tf.reset_default_graph()

    model = Sequential()
    model.add(GRU(units=512,
                  return_sequences=True,
                  input_shape=(1, 5)))
    model.add(Dropout(0.2))
    model.add(GRU(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

    # model = load_model("{}.h5".format(model_name))
    # print("MODEL-LOADED")

    model.fit(X_train,y_train,batch_size=250, epochs=500, validation_split=0.1, verbose=1)
    model.save("{}.h5".format(model_name))
    print('MODEL-SAVED')
else:
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