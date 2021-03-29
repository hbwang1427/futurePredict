import pandas as pd
import numpy as np

from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.callbacks import TensorBoard
#from Informer2020.exp.exp_informer import Exp_Informer

class NNmodel:

    def __init__(self,input_shape):
        print("cnn model is called")
        self.input_shape = input_shape

    def RNN(self):
        model = Sequential()
        model.add(GRU(units=512, return_sequences=True, input_shape=(1, self.input_shape)))
        model.add(Dropout(0.2))
        model.add(GRU(units=256))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid',kernel_initializer='random_normal', bias_initializer='zeros'))
        model.compile(loss='mse', optimizer='adam')

        return model
    
    def LSTM(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(1, self.input_shape)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25, activation='sigmoid',kernel_initializer='random_normal',bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid',kernel_initializer='random_normal',bias_initializer='zeros'))
        model.compile(loss='mse',optimizer='adam')
        return model

    def Informer(self):
        self.model = 'informer'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 3
        self.d_layers = 2
        self.d_ff = 1024
        self.factor = 5
        self.distil = True
        self.dropout = 0.05
        self.attn = 'prob'
        self.embed = 'timeF' ##
        self.activation = 'gelu'
        self.output_attension = True
        self.num_workers = 0
        self.train_epochs = 6
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'test'
        self.loss = 'mse'
        self.lradj = 'type1'
      
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_dt{}_{}_{}'.format(self.model, self.data, self.features, 
                self.seq_len, self.label_len, self.pred_len,
                self.d_model, self.n_heads, self.e_layers, self.d_layers, self.d_ff, self.attn, self.embed, self.distil, self.des, 1)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
