import numpy as np
from tensorflow import keras
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Bidirectional, Dense, LSTM, Add, Concatenate
from ._layers import DropoutLayer


class BiLSTM(Model):
    def __init__(self, train_data_shape, number_of_classes, number_of_layers=3, lstm_units=[32, 32, 32], drp=0.5,
                 dense_layers=1, dense_units=[128]):
        super(BiLSTM, self).__init__()
        self.num_of_classes = number_of_classes
        self.lstm_units = lstm_units
        self.n_layers = number_of_layers
        self.data_shape = train_data_shape
        self.dropout_percent = drp
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.model_name = 'BiLSTM'
        self.bilstm1 = Bidirectional(LSTM(units=self.lstm_units[0], activation="tanh", return_sequences=True))
        self.drp = DropoutLayer(self.dropout_percent)
        self.bilstm_block = BiLSTMBlock(self.n_layers, self.lstm_units, activation="tanh", drp_rate=self.dropout_percent)
        self.dense_block = DenseBlock(dense_layers, dense_units, activation="relu")
        self.out = Dense(self.num_of_classes, activation="softmax")

    def call(self, inputs):
        x = self.bilstm1(inputs)
        x = self.drp(x)
        x = self.bilstm_block(x)
        x = self.dense_block(x)
        x = self.out(x)

        return x

    def build_graph(self):
        x = Input(shape=self.data_shape)
        return Model(inputs=[x], outputs=self.call(x))


class BiLSTMBlock(Layer):
    def __init__(self, n_layers, lstm_units, activation="tanh", drp_rate=0.5):
        super(BiLSTMBlock, self).__init__()
        self.bilstm_list = list()
        for layer in range(0, n_layers - 1):
            bilstm = LSTM(units=lstm_units[layer], return_sequences=True, activation=activation)
            self.bilstm_list.append(bilstm)
            if layer == 0:
                self.bilstm_list.append(DropoutLayer(drp_rate))
        bilst = LSTM(units=lstm_units[layer], return_sequences=False, activation=activation)
        self.bilstm_list.append(bilst)

    def call(self, inputs):
        x = inputs
        for bilstm in self.bilstm_list:
            x = bilstm(x)

        return x


class DenseBlock(Layer):
    def __init__(self, n_layers, dense_units, activation="relu"):
        super(DenseBlock, self).__init__()
        self.dense_list = list()
        for d in range(0, n_layers):
            dense = Dense(units=dense_units[d], activation=activation)
            self.dense_list.append(dense)

    def call(self, inputs):
        x = inputs
        for dense in self.dense_list:
            x = dense(x)
        return x