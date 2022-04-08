import pandas as pd
import pydotplus
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.callbacks import TensorBoard,EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from file_load import load_future, series_to_supervised
import os
import argparse
from model_select import NNmodel

####input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--market', default='IF', help='market name (IC, IF, IH).')
parser.add_argument('-n', '--nSample', type=int, default=1, help='tick sampling for prediction.')
parser.add_argument('-s', '--startTime', type=int, default=20101020, help='start time for data collection')
parser.add_argument('-f', '--fTicks', type=int, default=20, help='future ticks to be predicted on.')
parser.add_argument('-g', '--gpuID', default='0', help='GPU ID.')
parser.add_argument('-r', '--retrain', default='Yes', help='retrain the model or not (Yes or No)')
parser.add_argument('--model', default='LSTM', help='select a model - RNN, LSTM or Informer')
parser.add_argument('--batchSize', type=int, default=1024, help='select a batch size for model training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for model training')

args = parser.parse_args()

root_path = 'StockFutureData'
market = args.market
nSample = args.nSample
startTime = args.startTime
fTicks = args.fTicks
gpuID = args.gpuID
epochs = args.epochs
batchSize = args.batchSize
model_name = 'future_price_'+args.model+'_'+market+'_'+str(nSample)+'_'+str(fTicks)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID # first gpu

print('start loading...')
mypath = os.path.join(root_path, market)
prices,label = load_future(mypath,startTime,nSample,fTicks)
print('done')

# conversion to numpy array
x = np.asarray(prices)
y = np.asarray(label)
x = np.vstack(x)
y = np.hstack(y)
print(x.shape,y.shape)

# scaling values for model
x_scale = MinMaxScaler(feature_range=(0,1))
y_scale = MinMaxScaler(feature_range=(0,1))

Xorig = x_scale.fit_transform(x)
Yorig = y_scale.fit_transform(y.reshape(-1,1))

X,Y = series_to_supervised(Xorig,Yorig,5,100)
nnum,steps,dim = X.shape

# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
#X_train = X_train.reshape((-1,dim*steps,1))
#X_test = X_test.reshape((-1,dim*steps,1))
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

if os.path.exists(model_name+'.h5'):
  mymodel = keras.models.load_model(model_name+'.h5')
else:
  NNmymodel = NNmodel(steps,dim)
  if args.model == 'GRU':
    mymodel = NNmymodel.GRU()
  elif args.model == 'LSTM':
    mymodel = NNmymodel.LSTM()
  elif args.model == 'Transformer':
    mymodel = NNmymodel.Transformer()
  mymodel.compile(loss='mse',optimizer='adam')

  tbCallBack = TensorBoard(log_dir=model_name+'_log',histogram_freq=0,write_graph=True,write_images=True)
  eStopCallback = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

  history = mymodel.fit(X_train, y_train, batchSize, epochs, validation_data=(X_test, y_test), verbose=1, callbacks=[tbCallBack, eStopCallback])
  mymodel.save("{}.h5".format(model_name))
  mymodel.save(model_name+'_model')
  print('MODEL-SAVED')
  
  # Plot training & validation loss values
  fig, ax = plt.subplots(figsize=(5, 5), sharex=True)
  plt.plot(history.history["loss"])
  plt.title("Model loss")
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
  plt.legend(["Train", "Test"], loc="upper left")
  plt.grid()
  plt.show()

score = mymodel.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = mymodel.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
plt.plot(yhat, label='Predicted')
plt.plot(y_test, label='Ground Truth')
plt.savefig(model_name+'.png')
plt.legend()
plt.show()
