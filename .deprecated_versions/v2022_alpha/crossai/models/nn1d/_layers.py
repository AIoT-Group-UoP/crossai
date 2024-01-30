from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, Dropout, Flatten


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCSpatialDropout1D(keras.layers.SpatialDropout1D):
    def call(self, inputs):
        return super().call(inputs, training=True)


class DropoutLayer(Layer):
    """
    This class creates a Droupout layer for a model.
    """
    def __init__(self, drp_rate=0.1, spatial=True):
        super(DropoutLayer, self).__init__()
        self.drp_rate = drp_rate
        self.spatial = spatial
        if spatial is True:
            self.drp = MCSpatialDropout1D(drp_rate)
        else:
            self.drp = MCDropout(drp_rate)

    def call(self, inputs):
        return self.drp(inputs)


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

    x = Flatten()(x)

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

    return x
