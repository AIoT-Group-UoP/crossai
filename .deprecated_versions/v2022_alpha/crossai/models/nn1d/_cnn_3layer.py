from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dense
from tensorflow.keras.layers import SeparableConv2D, MaxPooling1D, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from ._layers import dense_drop_block


class CNN3Layer:
    def __init__(self, input_shape,
                 num_classes,
                 kernel_initialize="he_uniform",
                 kernel_regularize=1e-5,
                 kernel_constraint=3,
                 dense_layers=1,
                 dense_units=[64],
                 dropout=True,
                 dropout_first=False,
                 dropout_rate=[0.7],
                 activation="softmax"):
        super(CNN3Layer, self).__init__()
        self._input_shape = input_shape
        self.num_classes = num_classes
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.dropout = dropout
        self.dropout_first = dropout_first
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build_graph(self):
        kernel_regularize = None
        if self.kernel_regularize is not None:
            kernel_regularize = l2(self.kernel_regularize)
        kernel_constraint = None
        if self.kernel_constraint is not None:
            kernel_constraint = MaxNorm(max_value=self.kernel_constraint, axis=[0, 1])

        # Create the input vector
        inputs = Input(shape=self._input_shape)

        x = Conv1D(filters=16, kernel_size=3, activation='relu',
                   kernel_initializer=self.kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(inputs)
        x = MaxPooling1D(2)(x)

        x = Conv1D(filters=32, kernel_size=3, activation='relu',
                   kernel_initializer=self.kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(x)
        x = MaxPooling1D(2)(x)

        x = Conv1D(filters=64, kernel_size=3, activation='relu',
                   kernel_initializer=self.kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(x)
        x = MaxPooling1D(2)(x)

        x = dense_drop_block(inputs=x, n_layers=self.dense_layers, dense_units=self.dense_units, dropout=self.dropout,
                             drop_first=self.dropout_first, drop_rate=self.dropout_rate, activation_dense="relu",
                             kernel_initialize=self.kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint)

        outputs = Dense(self.num_classes, activation=self.activation,
                        kernel_initializer=self.kernel_initialize,
                        kernel_regularizer=kernel_regularize,
                        kernel_constraint=kernel_constraint
                        )(x)

        model = Model(inputs, outputs)

        return model
