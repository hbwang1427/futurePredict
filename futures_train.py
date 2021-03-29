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
from model_select import NNmodel

####input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--market', default='IC', help='market name (IC, IF, IH).')
parser.add_argument('-n', '--nSample', type=int, default='5', help='tick sampling for prediction.')
parser.add_argument('-f', '--fTicks', type=int, default='20', help='future ticks to be predicted on.')
parser.add_argument('-g', '--gpuID', default='0', help='GPU ID.')
parser.add_argument('-r', '--retrain', default='Yes', help='retrain the model or not (Yes or No)')

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
prices,label = load_future(mypath,nSample,fTicks)

# conversion to numpy array
x, y = prices.values, label.values
nnum,dim = x.shape

# scaling values for model
x_scale = MinMaxScaler()
y_scale = MinMaxScaler()

X = x_scale.fit_transform(x)
Y = y_scale.fit_transform(y.reshape(-1,1))

# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01)
X_train = X_train.reshape((-1,1,dim))
X_test = X_test.reshape((-1,1,dim))

if not os.path.exists(model_name_h5) or args.retrain is 'Yes':
    # creating model using Keras
    # tf.reset_default_graph()

    NNmymodel = NNmodel(dim)
    mymodel = NNmymodel.RNN()

    # model = load_model("{}.h5".format(model_name))
    # print("MODEL-LOADED")
    tbCallBack = TensorBoard(log_dir=model_name,histogram_freq=0,write_graph=True,write_images=True)

    mymodel.fit(X_train,y_train,batch_size=512, epochs=100, validation_split=0.2, verbose=1, callbacks=[tbCallBack])
    mymodel.save("{}.h5".format(model_name))
    print('MODEL-SAVED')
else:
    mymodel = load_model(model_name_h5)

score = mymodel.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = mymodel.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
plt.plot(yhat, label='Predicted')
plt.plot(y_test, label='Ground Truth')
plt.savefig(model_name+'.png')
plt.legend()
plt.show()
