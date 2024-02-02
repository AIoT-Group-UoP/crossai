import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten


class DenseDropBlock(tf.keras.Model):
    def __init__(self, n_layers, dense_units, drop_rate, drop_first=False,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None,
                 activation="relu"):
        super().__init__()
        self.layer_list = list()
        for d in range(0, n_layers):
            drop = tf.keras.layers.Dropout(drop_rate[d])
            dense = tf.keras.layers.Dense(units=dense_units[d], 
                                          activation=activation,
                                          kernel_initializer=kernel_initialize,
                                          kernel_regularizer=kernel_regularize,
                                          kernel_constraint=kernel_constraint
                                          )
            if drop_first:
                self.layer_list.append(drop)
                self.layer_list.append(dense)
            else:
                self.layer_list.append(dense)
                self.layer_list.append(drop)

    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x


class DenseBlock(tf.keras.Model):
    def __init__(self, n_layers, dense_units,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None,
                 activation="relu"):
        super(DenseBlock, self).__init__()

        self.dense_list = list()
        for d in range(0, n_layers):
            dense = tf.keras.layers.Dense(units=dense_units[d], 
                                          activation=activation, 
                                          kernel_initializer=kernel_initialize,
                                          kernel_regularizer=kernel_regularize,
                                          kernel_constraint=kernel_constraint
                                          )
            self.dense_list.append(dense)

    def call(self, inputs):
        x = inputs
        for dense in self.dense_list:
            x = dense(x)
        return x


def dense_drop_block(inputs, n_layers, dense_units, dropout, drop_rate, drop_first=False, activation_dense="relu",
                     kernel_initialize=None,
                     kernel_regularize=None,
                     kernel_constraint=None
                     ):
    """A layer block that can initialize a series of Dropout/Dense, Dense/Dropout, or Dense layers.

    Args:
        inputs:
        n_layers:
        dense_units:
        dropout:
        drop_rate:
        drop_first:
        activation_dense:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    x = inputs
    if dropout:
        for d in range(0, n_layers):
            if drop_first:
                x = Dropout(drop_rate[d])(x)
                x = Dense(units=dense_units[d],
                          kernel_initializer=kernel_initialize,
                          kernel_regularizer=kernel_regularize,
                          kernel_constraint=kernel_constraint, activation=activation_dense
                          )(x)
            else:
                x = Dense(units=dense_units[d],
                          kernel_initializer=kernel_initialize,
                          kernel_regularizer=kernel_regularize,
                          kernel_constraint=kernel_constraint,
                          activation=activation_dense
                          )(x)
                x = Dropout(drop_rate[d])(x)
    else:
        for d in range(0, n_layers):
            x = Dense(units=dense_units[d],
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      activation=activation_dense
                      )(x)

    x = Flatten()(x)

    return x
