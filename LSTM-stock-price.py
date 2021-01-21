
# coding: utf-8

# In[37]:


import math
import datetime
import numpy as np
import pandas as pd
import tushare as ts
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler


# In[38]:


TRACK_DAYS=30


# In[39]:


pro = ts.pro_api('80bf1d654a719323377217d966da231f70950a88d8e1f18836865d6c')


# In[66]:


df = pro.daily(ts_code='000807.SZ', start_date='20030901', end_date=datetime.datetime.now().strftime("%Y%m%d"))
df = df.set_index('trade_date').sort_index(axis=0 ,ascending=True)


# In[67]:


plt.figure(figsize=(16,8))
plt.plot(df['close'])
plt.xlabel('trade_date', fontsize=18)
plt.show()


# In[68]:


dataset = df.filter(['close']).values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
training_data_len = math.ceil(len(scaled_data) * .8)
training_data = scaled_data[:training_data_len]

X_train = []
y_train = []
for i in np.arange(TRACK_DAYS, len(training_data)):
    X_train.append(training_data[i-TRACK_DAYS:i, 0])
    y_train.append(training_data[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)


# In[69]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=1, epochs=10)


# In[70]:


test_data = scaled_data[training_data_len-TRACK_DAYS:,:]

X_test = []
y_test = scaled_data[training_data_len:,:]
for i in np.arange(TRACK_DAYS,len(test_data)):
    X_test.append(test_data[i-TRACK_DAYS:i,0])
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#predictions


# In[71]:


data = df.filter(['close'])
train = data[:training_data_len]
valid = data[training_data_len:]
valid["prediction"] = predictions

#predictions

#valid
#valid[['close', 'prediction']]
plt.figure(figsize=(16,8))
plt.title('000807')
plt.plot(train[-10:]['close'])
plt.plot(valid[['close', 'prediction']])
plt.xlabel('trade_date', fontsize=18)
plt.ylabel('price', fontsize=18)
# for tick in plt.get_xticklabels():
#     tick.set_rotation(55)
plt.legend(['train', 'valid', 'prediction'])
plt.show()


# fig, ax = plt.subplots(tight_layout=True)
# #ax.plot(train[-10:]['close'])
# ax.plot(train['close'])
# ax.plot(valid[['close', 'prediction']])
# for tick in ax.get_xticklabels():
#     tick.set_rotation(55)
# ax.format_xdata = mdates.DateFormatter('%Y%m%d')
# plt.show()


# In[72]:


#forecast
X_test = scaled_data[-TRACK_DAYS:,0]
X_test = np.array(X_test)

X_test = X_test.reshape(1, X_test.shape[0], 1)
#X_test

y_pred = []
for i in range(200):
    pred_next = model.predict(X_test)
    #print(pred_next)
    X_test = np.append(X_test, [pred_next], axis=1)
    X_test = np.delete(X_test, 0, 1)
    y_pred.append(pred_next[0])

#y_pred
y_pred = scaler.inverse_transform(np.array(y_pred))
y_pred

