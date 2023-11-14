from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, \
    GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D, Flatten, concatenate, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from crossai.ai import dense_drop_block


# Implementation of InceptionV1 NN model based on:
# - https://arxiv.org/pdf/1409.4842v1.pdf
# InceptionV2-3: https://arxiv.org/pdf/1512.00567v3.pdf
# InceptionV4-ResNet: https://arxiv.org/pdf/1602.07261.pdf
def InceptionV1(input_shape,
                include_top=True,
                num_classes=1,
                classifier_activation="softmax",
                kernel_initialize=glorot_uniform(),
                kernel_regularize=1e-5,
                kernel_constraint=3,
                dense_layers=0,
                dense_units=[128, 128],
                dropout=False,
                dropout_first=False,
                dropout_rate=[0.5, 0.5],
                spatial=False,
                mc_inference=None
                ):
    """
    Builds the InceptionV1 or GoogLeNet Model.

    Args:
        input_shape (tuple): Shape of a single instance of the dataset.
        include_top (bool): If true, includes the fully-connected layer at the
            top.
        num_classes (int): Number of classes for prediction. Defaults to 1.
        classifier_activation (Union[str, Callable]): Activation function for
            the classification task.
        kernel_initialize (str): Kernel initializer. Defaults to
            "glorot uniform".
        kernel_regularize (Union[float, str]): Regularizer for the kernel, can
            be float or string. Default is None.
        kernel_constraint (int): Constraint on the kernel values. Default is 3.
        dropout (bool): If true, uses dropout. Defaults to False.
        dropout_first (bool): If true, adds dropout before the dense layer.
            Defaults to False.
        dropout_rate (list of float): Dropout rates for each layer.
        spatial (bool): If true, uses SpatialDropout2D, else Monte Carlo
            Dropout. Defaults to False.
        mc_inference (bool or None): Dropout setting during inference. True for
            enabled, False for disabled, None for training only. Defaults to
            None.

    Returns:
        keras.Model: An instance of the Keras Model.
    """

    # Initializer - regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    bias_initialize = Constant(value=0.2)

    # -- Initiating Model Topology --
    # Create the input vector
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Conv2D(64, (7, 7), padding="same", strides=(2, 2), activation="relu",
               name="conv_1_7x7/2", kernel_initializer=kernel_initialize,
               bias_initializer=bias_initialize)(input_layer)
    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_1_3x3/2")(x)
    x = Conv2D(64, (1, 1), padding="same", strides=(1, 1), activation="relu",
               name="conv_2a_3x3/1")(x)
    x = Conv2D(192, (3, 3), padding="same", strides=(1, 1), activation="relu",
               name="conv_2b_3x3/1")(x)
    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_2_3x3/2")(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_3a")

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_3b")

    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_3_3x3/2")(x)

    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4a")

    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding="same", activation="relu")(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation="relu")(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(10, activation="softmax", name="auxilliary_output_1")(x1)

    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4b")

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4c")

    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4d")

    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding="same", activation="relu")(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation="relu")(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(10, activation="softmax", name="auxilliary_output_2")(x2)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4e")

    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_4_3x3/2")(x)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_5a")

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_5b")

    if include_top:
        # flatten
        # x = Flatten()(x)

        x = GlobalAveragePooling2D(name="avg_pool_5_3x3/1")(x)

        # apply multiple sequential dense/dropout layers
        x = dense_drop_block(inputs=x,
                             n_layers=dense_layers,
                             dense_units=dense_units,
                             dropout=dropout,
                             drop_first=dropout_first,
                             drop_rate=dropout_rate,
                             activation_dense="relu",
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint,
                             spatial=spatial,
                             mc_inference=mc_inference
                             )

        # Fully connected output layer (classification)
        x = Dense(num_classes, activation=classifier_activation,
                  name="output")(x)

    model = Model(input_layer, [x, x1, x2], name="inception_v1")

    return model


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     kernel_initialize,
                     kernel_regularize,
                     kernel_constraint,
                     bias_initialize,
                     name=None):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(x)

    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(x)

    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding="same")(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding="same",
                       activation="relu",
                       kernel_initializer=kernel_initialize,
                       kernel_regularizer=kernel_regularize,
                       kernel_constraint=kernel_constraint,
                       bias_initializer=bias_initialize)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj],
                         axis=3,
                         name=name)

    return output
