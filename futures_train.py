import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from file_load import load_future, load_Daily
import os
import argparse

####input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--market', help='market name (IC, IF, IH).')
parser.add_argument('-n', '--nSample', type=int, help='tick sampling for prediction.')
parser.add_argument('-f', '--fTicks', type=int, help='future ticks to be predicted on.')
parser.add_argument('-g', '--gpuID', help='GPU ID.')
parser.add_argument('-r', '--retrain', help='retrain the model or not (Yes or No)')

args = parser.parse_args()

root_path = 'StockFutureData'
market = args.market
nSample = args.nSample
fTicks = args.fTicks
gpuID = args.gpuID
model_name = 'future_price_GRU'+'_'+market+'_'+str(nSample)+'_'+str(fTicks)
model_name_h5= model_name+'.h5'

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID # first gpu

mypath = os.path.join(root_path, market)
prices = load_future(mypath,nSample,fTicks)

# conversion to numpy array
x, y = prices.values, label.values

# scaling values for model
x_scale = MinMaxScaler()
y_scale = MinMaxScaler()

X = x_scale.fit_transform(x)
Y = y_scale.fit_transform(y.reshape(-1,1))

# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.001)
X_train = X_train.reshape((-1,1,14))
X_test = X_test.reshape((-1,1,14))

if not os.path.exists(model_name_h5) or args.retrain is 'Yes':
    # creating model using Keras
    # tf.reset_default_graph()

    model = Sequential()
    model.add(GRU(units=512,
                  return_sequences=True,
                  input_shape=(1, 14)))
    #model.add(Dropout(0.2))
    model.add(GRU(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

    # model = load_model("{}.h5".format(model_name))
    # print("MODEL-LOADED")
    tbCallBack = TensorBoard(log_dir=model_name,histogram_freq=0,write_graph=True,write_images=True,update_freq="epoch")

    model.fit(X_train,y_train,batch_size=250, epochs=500, validation_split=0.1, verbose=1, callbacks=[tbCallBack])
    model.save("{}.h5".format(model_name))
    print('MODEL-SAVED')
else:
    model = load_model(model_name_h5)

score = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
plt.plot(yhat, label='Predicted')
plt.plot(y_test, label='Ground Truth')
plt.savefig(model_name+'.png')
plt.legend()
plt.show()
