# This architecture is based on ResNet 34 (2015)
# Paper: https://arxiv.org/pdf/1512.03385.pdf

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ReLU, BatchNormalization
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from ._layers import DenseBlock, DenseDropBlock


class ResNet34(tf.keras.Model):
    def __init__(self, input_shape,
                 num_classes,
                 kernel_initialize="he_normal",
                 kernel_regularize=1e-3,
                 kernel_constraint=3,
                 pooling_type="avg",
                 dense_layers=0,
                 dense_units=[128, 128],
                 dropout=False,
                 dropout_first=False,
                 dropout_rate=[0.5, 0.5],
                 activation="softmax"):
        """

        Args:
            input_shape:
            num_classes:
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:
            pooling_type (str, None): (default: "avg")
            dense_layers:
            dense_units:
            dropout:
            dropout_first:
            dropout_rate:
            activation:
        """
        super(ResNet34, self).__init__()
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

        if self.kernel_regularize is not None:
            self.kernel_regularize = l2(self.kernel_regularize)
        if self.kernel_constraint is not None:
            self.kernel_constraint = MaxNorm(max_value=self.kernel_constraint, axis=[0, 1])

        # The Stem Convolution Group
        self.stem = Stem(kernel_initialize=self.kernel_initialize,
                         kernel_regularize=self.kernel_regularize,
                         kernel_constraint=self.kernel_constraint
                         )

        # The learner
        self.learner = Learner(kernel_initialize=self.kernel_initialize,
                               kernel_regularize=self.kernel_regularize,
                               kernel_constraint=self.kernel_constraint
                               )

        # The Classifier for X classes
        self.outputs = Classifier(num_classes=self.num_classes,
                                  kernel_initialize=self.kernel_initialize,
                                  kernel_regularize=self.kernel_regularize,
                                  kernel_constraint=self.kernel_constraint,
                                  pooling_type=self.pooling_type,
                                  dense_layers=self.dense_layers,
                                  dense_units=self.dense_units,
                                  dropout=self.dropout,
                                  dropout_first=self.dropout_first,
                                  dropout_rate=self.dropout_rate,
                                  activation=self.activation
                                  )

    def call(self, inputs):
        x = self.stem(inputs)
        x = self.learner(x)
        x = self.outputs(x)
        return x

    def build_graph(self):
        # input_shape=(224, 224, 3)
        x = tf.keras.Input(shape=self._input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Stem(tf.keras.layers.Layer):
    """ Construct the Stem Convolution Group
        inputs : input vector
    """
    def __init__(self,
                 kernel_initialize="he_normal",
                 kernel_regularize=1e-3,
                 kernel_constraint=3,
                 ):
        super(Stem, self).__init__()
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        # First Convolutional layer, where pooled feature maps will be reduced by 75%
        self.conv2d = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu',
                             kernel_initializer=self.kernel_initialize,
                             kernel_regularizer=self.kernel_regularize,
                             kernel_constraint=self.kernel_constraint
                             )
        self.max_pool2d = MaxPooling2D((3, 3), strides=(2, 2), padding='same')

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.max_pool2d(x)
        return x


class Learner(tf.keras.layers.Layer):
    """ Construct the Learner
        x  : input to the learner
    """
    def __init__(self,
                 kernel_initialize="he_normal",
                 kernel_regularize=1e-3,
                 kernel_constraint=3,
                 ):
        super(Learner, self).__init__()
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        # First Residual Block Group of 64 filters
        self.res_group_0 = ResidualGroup(n_filters=64, n_blocks=3,
                                         kernel_initialize=self.kernel_initialize,
                                         kernel_regularize=self.kernel_regularize,
                                         kernel_constraint=self.kernel_constraint
                                         )
        # Second Residual Block Group of 128 filters
        self.res_group_1 = ResidualGroup(n_filters=128, n_blocks=3,
                                         kernel_initialize=self.kernel_initialize,
                                         kernel_regularize=self.kernel_regularize,
                                         kernel_constraint=self.kernel_constraint
                                         )
        # Third Residual Block Group of 256 filters
        self.res_group_2 = ResidualGroup(n_filters=256, n_blocks=5,
                                         kernel_initialize=self.kernel_initialize,
                                         kernel_regularize=self.kernel_regularize,
                                         kernel_constraint=self.kernel_constraint
                                         )
        # Fourth Residual Block Group of 512 filters
        self.res_group_3 = ResidualGroup(n_filters=512, n_blocks=2, conv=False,
                                         kernel_initialize=self.kernel_initialize,
                                         kernel_regularize=self.kernel_regularize,
                                         kernel_constraint=self.kernel_constraint
                                         )

    def call(self, inputs):
        x = self.res_group_0(inputs)
        x = self.res_group_1(x)
        x = self.res_group_2(x)
        x = self.res_group_3(x)
        return x


class ResidualGroup(tf.keras.layers.Layer):
    def __init__(self, n_filters, n_blocks, conv=True,
                 kernel_initialize="he_normal",
                 kernel_regularize=1e-3,
                 kernel_constraint=3,
                 ):
        """ Construct a Residual Group
            n_filters: number of filters
            n_blocks : number of blocks in the group
            conv     : flag to include the convolution block connector
        """
        super(ResidualGroup, self).__init__()
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.conv = conv
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        self.res_block = ResidualBlock(n_filters=n_filters,
                                       kernel_initialize=self.kernel_initialize,
                                       kernel_regularize=self.kernel_regularize,
                                       kernel_constraint=self.kernel_constraint
                                       )

        # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
        self.conv_blck = ConvBlock(n_filters=n_filters * 2,
                                   kernel_initialize=self.kernel_initialize,
                                   kernel_regularize=self.kernel_regularize,
                                   kernel_constraint=self.kernel_constraint
                                   )

    def call(self, inputs):

        for _ in range(self.n_blocks):
            x = self.res_block(inputs)

        if self.conv:
            x = self.conv_blck(x)

        return x


class ResidualBlock(tf.keras.layers.Layer):
    """ Construct a Residual Block of Convolutions
        x        : input into the block
        n_filters: number of filters
    """
    def __init__(self, n_filters,
                 kernel_initialize="he_normal",
                 kernel_regularize=1e-3,
                 kernel_constraint=3,
                 ):
        super(ResidualBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        self.conv2d_0 = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", activation="relu",
                               kernel_initializer=self.kernel_initialize,
                               kernel_regularizer=self.kernel_regularize,
                               kernel_constraint=self.kernel_constraint
                               )
        self.conv2d_1 = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", activation="relu",
                               kernel_initializer=self.kernel_initialize,
                               kernel_regularizer=self.kernel_regularize,
                               kernel_constraint=self.kernel_constraint
                               )
        self.add = Add()

    def call(self, inputs):
        shortcut = inputs
        x = self.conv2d_0(inputs)
        x = self.conv2d_1(x)
        x = self.add([shortcut, x])
        return x


class ConvBlock(tf.keras.layers.Layer):
    """ Construct Block of Convolutions without Pooling
        x        : input into the block
        n_filters: number of filters
    """
    def __init__(self, n_filters,
                 kernel_initialize="he_normal",
                 kernel_regularize=1e-3,
                 kernel_constraint=3,
                 ):
        super(ConvBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        self.conv2d_0 = Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same", activation="relu",
                               kernel_initializer=self.kernel_initialize,
                               kernel_regularizer=self.kernel_regularize,
                               kernel_constraint=self.kernel_constraint)
        self.conv2d_1 = Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same", activation="relu",
                               kernel_initializer=self.kernel_initialize,
                               kernel_regularizer=self.kernel_regularize,
                               kernel_constraint=self.kernel_constraint)

    def call(self, inputs):
        x = self.conv2d_0(inputs)
        x = self.conv2d_1(x)
        return x


class Classifier(tf.keras.layers.Layer):
    """ Construct the Classifier Group
        x         : input vector
        n_classes : number of output classes
    """
    def __init__(self, num_classes,
                 kernel_initialize="he_normal",
                 kernel_regularize=1e-3,
                 kernel_constraint=3,
                 pooling_type="avg",
                 dense_layers=0,
                 dense_units=[128, 128],
                 dropout=False,
                 dropout_first=False,
                 dropout_rate=[0.5, 0.5],
                 activation="softmax"
                 ):
        super(Classifier, self).__init__()
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

        # Pool at the end of all the convolutional residual blocks
        self.global_average_pool = GlobalAveragePooling2D()

        if self.dropout:
            self.dense_block = DenseDropBlock(n_layers=self.dense_layers,
                                              dense_units=self.dense_units,
                                              drop_rate=self.dropout_rate,
                                              drop_first=self.dropout_first,
                                              kernel_initialize=self.kernel_initialize,
                                              kernel_regularize=self.kernel_regularize,
                                              kernel_constraint=self.kernel_constraint,
                                              activation="relu"
                                              )
        else:
            self.dense_block = DenseBlock(n_layers=self.dense_layers,
                                          dense_units=self.dense_units,
                                          kernel_initialize=self.kernel_initialize,
                                          kernel_regularize=self.kernel_regularize,
                                          kernel_constraint=self.kernel_constraint,
                                          activation="relu"
                                          )
        self.flatten = Flatten()

        # Final Dense Outputting Layer for the outputs
        self.outputs = Dense(self.num_classes, activation=self.activation,
                             kernel_initializer=self.kernel_initialize,
                             kernel_regularizer=self.kernel_regularize,
                             kernel_constraint=self.kernel_constraint
                             )

    def call(self, inputs):
        x = inputs
        if self.pooling_type == "avg":
            x = self.global_average_pool(x)
        if self.dense_layers is not None or self.dense_layers != 0:
            x = self.dense_block(inputs)
        x = self.flatten(x)
        x = self.outputs(x)

        return x
