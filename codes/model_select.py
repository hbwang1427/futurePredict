import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, GRU
import pydot
import pydotplus
from pydotplus import graphviz

class NNmodel:

    def __init__(self,timesteps,feature_shape):
        print("cnn model is called")
        self.timesteps = timesteps
        self.feature_shape = feature_shape

    def GRU(self):
        model = Sequential()
        model.add(GRU(units=64, return_sequences=True, input_shape=(self.timesteps,self.feature_shape)))
        model.add(GRU(units=64,return_sequences=True))
        model.add(GRU(units=32
        ,return_sequences=True))
        model.add(GRU(units=32, dropout=0.2))
        model.add(Dense(1, activation='sigmoid',kernel_initializer='random_normal', bias_initializer='zeros'))
        model.summary()
        #tf.keras.utils.plot_model(model, to_file=os.path.join('./model_gru.png'), show_shapes=True, show_layer_names=True)
        return model
    
    def LSTM(self):
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(self.timesteps,self.feature_shape)))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(LSTM(units=32, dropout=0.2))
        model.add(Dense(units=1, activation='sigmoid',kernel_initializer='random_normal',bias_initializer='zeros'))
        model.summary()
        #tf.keras.utils.plot_model(model, to_file=os.path.join('./model_lstm.png'), show_shapes=True, show_layer_names=True)
        return model
    
    def __transformer_encoder__(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
    
        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    def Transformer(self, n_classes, input_shape, head_size, num_heads, ff_dim,
        num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.__transformer_encoder__(x, head_size, num_heads, ff_dim, dropout)
    
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        return keras.Model(inputs, outputs)

  
