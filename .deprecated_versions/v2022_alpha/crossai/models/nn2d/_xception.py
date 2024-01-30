# This architecture is based on Xception (2016)
# https://arxiv.org/pdf/1610.02357.pdf

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Add, Activation, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from ._layers import DenseBlock, DenseDropBlock


class Xception(tf.keras.Model):
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
        super(Xception, self).__init__()
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

        self.entry_flow = EntryFlow(kernel_initialize=self.kernel_initialize,
                                    kernel_regularize=self.kernel_regularize,
                                    kernel_constraint=self.kernel_constraint
                                    )

        self.middle_flow = MiddleFlow(kernel_initialize=self.kernel_initialize,
                                      kernel_regularize=self.kernel_regularize,
                                      kernel_constraint=self.kernel_constraint
                                      )

        self.exit_flow = ExitFlow(num_classes=self.num_classes,
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

        x = self.entry_flow(inputs)

        x = self.middle_flow(x)

        x = self.exit_flow(x)

        return x

    def build_graph(self):
        # input_shape=(224, 224, 3)
        x = tf.keras.Input(shape=self._input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class EntryFlow(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None
                 ):
        """

        Args:
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:
        """
        super(EntryFlow, self).__init__()
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        self.stem = Stem(kernel_initialize=self.kernel_initialize,
                         kernel_regularize=self.kernel_regularize,
                         kernel_constraint=self.kernel_constraint)

    def call(self, inputs):
        x = self.stem(inputs)

        # for n_filters in [128, 256, 728]:
        #     x = ProjectionBlock(n_filters=n_filters,
        #                         kernel_initialize=self.kernel_initialize,
        #                         kernel_regularize=self.kernel_regularize,
        #                         kernel_constraint=self.kernel_constraint
        #                         )(x)
        return x


class MiddleFlow(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None
                 ):
        """Create the middle flow section

        Args:
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:
        """
        super(MiddleFlow, self).__init__()
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        # instantiate a residual block
        self.residual_block = ResidualBlock(n_filters=728,
                                            kernel_initialize=self.kernel_initialize,
                                            kernel_regularize=self.kernel_regularize,
                                            kernel_constraint=self.kernel_constraint
                                            )

    def call(self, inputs):
        # Create 8 residual blocks
        for _ in range(8):
            x = self.residual_block(inputs)

        return x


class ExitFlow(tf.keras.layers.Layer):
    def __init__(self, num_classes,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None,
                 pooling_type="avg",
                 dense_layers=0,
                 dense_units=[128, 128],
                 dropout=False,
                 dropout_first=False,
                 dropout_rate=[0.5, 0.5],
                 activation="softmax"
                 ):
        """Create the exit flow section

        Args:
            num_classes:
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:
            pooling_type:
            dense_layers:
            dense_units:
            dropout:
            dropout_first:
            dropout_rate:
            activation:
        """
        super(ExitFlow, self).__init__()
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

        # 1x1 strided convolution to increase number and reduce size of feature maps
        # in identity link to match output of residual block for the add operation (projection shortcut)
        self.conv2d_0 = Conv2D(1024, (1, 1), strides=(2, 2), padding='same')
        self.bn_0 = BatchNormalization()

        # First Depthwise Separable Convolution
        # Dimensionality reduction - reduce number of filters
        self.sep_conv2d_1 = SeparableConv2D(728, (3, 3), padding='same')
        self.bn_1 = BatchNormalization()
        self.relu_1 = Activation("relu")

        # Second Depthwise Separable Convolution
        # Dimensionality restoration
        self.sep_conv2d_2 = SeparableConv2D(1024, (3, 3), padding='same')
        self.bn_2 = BatchNormalization()
        self.relu_2 = Activation("relu")

        # Create pooled feature maps, reduce size by 75%
        self.max_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')

        # Add the projection shortcut to the output of the pooling layer
        self.add = Add()

        # Third Depthwise Separable Convolution
        self.sep_conv2d_3 = SeparableConv2D(1556, (3, 3), padding='same')
        self.bn_3 = BatchNormalization()
        self.relu_3 = Activation("relu")

        # Fourth Depthwise Separable Convolution
        self.sep_conv2d_4 = SeparableConv2D(2048, (3, 3), padding='same')
        self.bn_4 = BatchNormalization()
        self.relu_4 = Activation("relu")

        self.classifier = Classifier(num_classes=self.num_classes,
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

        shortcut = self.conv2d_0(inputs)
        shortcut = self.bn_0(shortcut)

        x = self.sep_conv2d_1(inputs)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.sep_conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.max_pool(x)

        x = self.add([x, shortcut])

        x = self.sep_conv2d_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        x = self.sep_conv2d_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)

        x = self.classifier(x)

        return x


class Stem(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None,
                 ):
        """Create the stem entry into the neural network

        Args:
            self:
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:

        Returns:

        """
        super(Stem, self).__init__()
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        # Strided convolution - dimensionality reduction
        # Reduce feature maps by 75%
        self.conv_2d_0 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2),
                                kernel_initializer=self.kernel_initialize,
                                kernel_regularizer=self.kernel_regularize,
                                kernel_constraint=self.kernel_constraint
                                )
        self.bn_0 = BatchNormalization()
        self.relu_0 = Activation("relu")

        # Convolution - dimensionality expansion
        # Double the number of filters
        self.conv_2d_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                kernel_initializer=self.kernel_initialize,
                                kernel_regularizer=self.kernel_regularize,
                                kernel_constraint=self.kernel_constraint
                                )
        self.bn_1 = BatchNormalization()
        self.relu_1 = Activation("relu")

    def call(self, inputs):
        # input tensor to neural network
        x = self.conv_2d_0(inputs)
        x = self.bn_0(x)
        x = self.relu_0(x)

        x = self.conv_2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        return x


class ProjectionBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None,
                 ):
        """Create a residual block using Depthwise Separable Convolutions with Projection shortcut

            Args:
                n_filters: number of filters
                kernel_initialize:
                kernel_regularize:
                kernel_constraint:
        """
        super(ProjectionBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        # Strided convolution to double number of filters in identity link to
        # match output of residual block for the add operation (projection shortcut)
        self.conv2d_0 = Conv2D(filters=self.n_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                               kernel_initializer=self.kernel_initialize,
                               kernel_regularizer=self.kernel_regularize,
                               kernel_constraint=self.kernel_constraint
                               )
        self.bn_0 = BatchNormalization()

        # First Depthwise Separable Convolution
        self.sep_conv2d_0 = SeparableConv2D(filters=self.n_filters, kernel_size=(3, 3), padding='same',
                                            depthwise_initializer=kernel_initialize,
                                            pointwise_initializer=kernel_initialize,
                                            depthwise_regularizer=kernel_regularize,
                                            pointwise_regularizer=kernel_regularize,
                                            depthwise_constraint=kernel_constraint,
                                            pointwise_constraint=kernel_constraint
                                            )
        self.bn_1 = BatchNormalization()
        self.relu_0 = Activation("relu")

        # Second depthwise Separable Convolution
        self.sep_conv2d_1 = SeparableConv2D(filters=n_filters, kernel_size=(3, 3), padding='same',
                                            depthwise_initializer=kernel_initialize,
                                            pointwise_initializer=kernel_initialize,
                                            depthwise_regularizer=kernel_regularize,
                                            pointwise_regularizer=kernel_regularize,
                                            depthwise_constraint=kernel_constraint,
                                            pointwise_constraint=kernel_constraint
                                            )
        self.bn_2 = BatchNormalization()
        self.relu_1 = Activation("relu")

        # Create pooled feature maps, reduce size by 75%
        self.max_pool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Add the projection shortcut to the output of the block
        self.add = Add()

    def call(self, inputs):
        # Remember the input
        shortcut = inputs
        x = inputs

        shortcut = self.conv2d_0(shortcut)
        shortcut = self.bn_0(shortcut)

        x = self.sep_conv2d_0(x)
        x = self.bn_1(x)
        x = self.relu_0(x)

        x = self.sep_conv2d_1(x)
        x = self.bn_2(x)
        x = self.relu_1(x)

        x = self.max_pool_0(x)

        x = self.add([x, shortcut])

        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None
                 ):
        """Create a residual block using Depthwise Separable Convolutions
            x = input into residual block
            Args:
                n_filters: number of filters
                kernel_initialize:
                kernel_regularize:
                kernel_constraint:
        """
        super(ResidualBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        # First Depthwise Separable Convolution
        self.sep_conv_0 = SeparableConv2D(filters=self.n_filters, kernel_size=(3, 3), padding='same',
                                          depthwise_initializer=kernel_initialize,
                                          pointwise_initializer=kernel_initialize,
                                          depthwise_regularizer=kernel_regularize,
                                          pointwise_regularizer=kernel_regularize,
                                          depthwise_constraint=kernel_constraint,
                                          pointwise_constraint=kernel_constraint
                                          )
        self.bn_0 = BatchNormalization()
        self.relu_0 = Activation("relu")

        # Second depthwise Separable Convolution
        self.sep_conv_1 = SeparableConv2D(filters=self.n_filters, kernel_size=(3, 3), padding='same',
                                          depthwise_initializer=kernel_initialize,
                                          pointwise_initializer=kernel_initialize,
                                          depthwise_regularizer=kernel_regularize,
                                          pointwise_regularizer=kernel_regularize,
                                          depthwise_constraint=kernel_constraint,
                                          pointwise_constraint=kernel_constraint
                                          )
        self.bn_1 = BatchNormalization()
        self.relu_1 = Activation("relu")

        # Third depthwise Separable Convolution
        self.sep_conv_2 = SeparableConv2D(filters=self.n_filters, kernel_size=(3, 3), padding='same',
                                          depthwise_initializer=kernel_initialize,
                                          pointwise_initializer=kernel_initialize,
                                          depthwise_regularizer=kernel_regularize,
                                          pointwise_regularizer=kernel_regularize,
                                          depthwise_constraint=kernel_constraint,
                                          pointwise_constraint=kernel_constraint
                                          )
        self.bn_2 = BatchNormalization()
        self.relu_2 = Activation("relu")

        # Add the identity link to the output of the block
        self.add = Add()

    def call(self, inputs):
        x = inputs
        # Remember the input
        shortcut = x

        x = self.sep_conv_0(x)
        x = self.bn_0(x)
        x = self.relu_0(x)

        x = self.sep_conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.sep_conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.add([x, shortcut])

        return x


class Classifier(tf.keras.layers.Layer):
    def __init__(self, num_classes,
                 kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None,
                 pooling_type="avg",
                 dense_layers=0,
                 dense_units=[128, 128],
                 dropout=False,
                 dropout_first=False,
                 dropout_rate=[0.5, 0.5],
                 activation="softmax"
                 ):
        """The output classifier

        Args:
            num_classes:
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:
            pooling_type:
            dense_layers:
            dense_units:
            dropout:
            dropout_first:
            dropout_rate:
            activation:
        """
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.activation = activation
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

        # Global Average Pooling will flatten the 10x10 feature maps into 1D
        # feature maps
        self.global_average_pool = GlobalAveragePooling2D()

        if self.dropout:
            self.dense_block = DenseDropBlock(n_layers=self.dense_layers,
                                              dense_units=self.dense_units,
                                              drop_rate=self.dropout_rate,
                                              activation="relu",
                                              drop_first=self.dropout_first
                                              )
        else:
            self.dense_block = DenseBlock(n_layers=self.dense_layers,
                                          dense_units=self.dense_units,
                                          activation="relu"
                                          )
        self.flatten = Flatten()

        # Fully connected output layer (classification)
        self.outputs = Dense(self.num_classes, activation=self.activation,
                             kernel_initializer=self.kernel_initialize,
                             kernel_regularizer=self.kernel_regularize,
                             kernel_constraint=self.kernel_constraint
                             )

    def call(self, inputs):
        # input to the classifier
        x = inputs
        if self.pooling_type == "avg":
            x = self.global_average_pool(x)
        if self.dense_layers is not None or self.dense_layers != 0:
            x = self.dense_block(inputs)
        x = self.outputs(x)

        return x
