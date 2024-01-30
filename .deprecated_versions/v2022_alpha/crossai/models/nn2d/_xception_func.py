import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from ._layers import dense_drop_block


class XceptionFunc:
    def __init__(self, input_shape,
                 num_classes,
                 kernel_initialize="he_uniform",
                 kernel_regularize=1e-5,
                 kernel_constraint=3,
                 pooling_type="avg",
                 dense_layers=0,
                 dense_units=[128, 128],
                 dropout=False,
                 dropout_first=False,
                 dropout_rate=[0.5, 0.5],
                 activation="softmax"):
        super(XceptionFunc, self).__init__()
        self._input_shape = input_shape
        self.num_classes = num_classes
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        self.pooling_type = pooling_type
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.dropout = dropout
        self.dropout_first = dropout_first
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build_graph(self):

        if self.kernel_regularize is not None:
            self.kernel_regularize = l2(self.kernel_regularize)
        if self.kernel_constraint is not None:
            self.kernel_constraint = MaxNorm(max_value=self.kernel_constraint, axis=[0, 1])

        # Create the input vector
        inputs = Input(shape=self._input_shape)

        # Create entry section
        x = entry_flow(inputs,
                       kernel_initialize=self.kernel_initialize,
                       kernel_regularize=self.kernel_regularize,
                       kernel_constraint=self.kernel_constraint
                       )

        # Create the middle section
        x = middle_flow(x,
                        kernel_initialize=self.kernel_initialize,
                        kernel_regularize=self.kernel_regularize,
                        kernel_constraint=self.kernel_constraint
                        )

        # Create the exit section for 2 classes
        outputs = exit_flow(x, self.num_classes,
                            pooling_type=self.pooling_type,
                            dense_layers=self.dense_layers,
                            dense_units=self.dense_units,
                            dropout=self.dropout,
                            dropout_first=self.dropout_first,
                            dropout_rate=self.dropout_rate,
                            kernel_initialize=self.kernel_initialize,
                            kernel_regularize=self.kernel_regularize,
                            kernel_constraint=self.kernel_constraint,
                            activation=self.activation
                            )

        model = Model(inputs, outputs)

        return model


def entry_flow(inputs, kernel_initialize, kernel_regularize, kernel_constraint):
    """ Create the entry flow section
        inputs : input tensor to neural network
    """

    def stem(inputs, kernel_initialize, kernel_regularize, kernel_constraint):
        """ Create the stem entry into the neural network
            inputs : input tensor to neural network
        """
        # Strided convolution - dimensionality reduction
        # Reduce feature maps by 75%
        x = Conv2D(32, (3, 3), strides=(2, 2),
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Convolution - dimensionality expansion
        # Double the number of filters
        x = Conv2D(64, (3, 3), strides=(1, 1),
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    # Create the stem to the neural network
    x = stem(inputs, kernel_initialize, kernel_regularize, kernel_constraint)

    # Create three residual blocks using linear projection
    for n_filters in [128, 256, 728]:
        x = projection_block(x, n_filters, kernel_initialize, kernel_regularize, kernel_constraint)

    return x


def middle_flow(x, kernel_initialize, kernel_regularize, kernel_constraint):
    """ Create the middle flow section
        x : input tensor into section
    """
    # Create 8 residual blocks
    for _ in range(8):
        x = residual_block(x, 728, kernel_initialize, kernel_regularize, kernel_constraint)
    return x


def exit_flow(x, n_classes, pooling_type, dense_layers, dense_units, dropout, dropout_first, dropout_rate,
              kernel_initialize, kernel_regularize, kernel_constraint,
              activation):
    """ Create the exit flow section
        x         : input to the exit flow section
        n_classes : number of output classes
    """

    def classifier(x, n_classes, pooling_type, dense_layers, dense_units, dropout, dropout_first, dropout_rate,
                   kernel_initialize, kernel_regularize, kernel_constraint,
                   activation
                   ):
        """ The output classifier
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Global Average Pooling will flatten the 10x10 feature maps into 1D
        # feature maps
        if pooling_type == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling_type is None:
            x = x
        else:
            raise Exception("Please define a valid pooling_type value: 'avg' or None")

        x = dense_drop_block(inputs=x, n_layers=dense_layers, dense_units=dense_units, dropout=dropout,
                             drop_first=dropout_first, drop_rate=dropout_rate, activation_dense="relu",
                             kernel_initialize=kernel_initialize, kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint)

        # Fully connected output layer (classification)
        x = Dense(n_classes, activation=activation)(x)

        return x

    # 1x1 strided convolution to increase number and reduce size of feature maps
    # in identity link to match output of residual block for the add operation (projection shortcut)
    shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='same',
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint
                      )(x)
    shortcut = BatchNormalization()(shortcut)

    # First Depthwise Separable Convolution
    # Dimensionality reduction - reduce number of filters
    x = SeparableConv2D(728, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second Depthwise Separable Convolution
    # Dimensionality restoration
    x = SeparableConv2D(1024, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add the projection shortcut to the output of the pooling layer
    x = Add()([x, shortcut])

    # Third Depthwise Separable Convolution
    x = SeparableConv2D(1556, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Fourth Depthwise Separable Convolution
    x = SeparableConv2D(2048, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create classifier section
    x = classifier(x=x, n_classes=n_classes, pooling_type=pooling_type, dense_layers=dense_layers,
                   dense_units=dense_units, dropout=dropout, dropout_rate=dropout_rate, dropout_first=dropout_first,
                   kernel_initialize=kernel_initialize, kernel_regularize=kernel_regularize,
                   kernel_constraint=kernel_constraint,
                   activation=activation)

    return x


def projection_block(x, n_filters, kernel_initialize, kernel_regularize, kernel_constraint):
    """ Create a residual block using Depthwise Separable Convolutions with Projection shortcut
        x        : input into residual block
        n_filters: number of filters
    """
    # Remember the input
    shortcut = x

    # Strided convolution to double number of filters in identity link to
    # match output of residual block for the add operation (projection shortcut)
    shortcut = Conv2D(n_filters, (1, 1), strides=(2, 2), padding='same',
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint
                      )(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # First Depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add the projection shortcut to the output of the block
    x = Add()([x, shortcut])

    return x


def residual_block(x, n_filters, kernel_initialize, kernel_regularize, kernel_constraint):
    """ Create a residual block using Depthwise Separable Convolutions
        x        : input into residual block
        n_filters: number of filters
    """
    # Remember the input
    shortcut = x

    # First Depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same',
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Add the identity link to the output of the block
    x = Add()([x, shortcut])
    return x
