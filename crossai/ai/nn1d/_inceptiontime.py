from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Add, Activation, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from .._layers_dropout import dropout_layer_1d


# Implementation of InceptionTime NN model based on:
# - https://arxiv.org/pdf/1909.04939
def InceptionTime(input_shape,
                  include_top=True,
                  num_classes=1,
                  classifier_activation="softmax",
                  nb_filters=32,
                  use_residual=True,
                  use_bottleneck=True,
                  depth=6,
                  kernel_size=41,
                  bottleneck_size=32,
                  drp_input=0,
                  drp_high=0,
                  kernel_initialize="he_uniform",
                  kernel_regularize=4e-5,
                  kernel_constraint=3,
                  spatial=False,
                  mc_inference=None):
    """An ensemble of deep Convolutional Neural Network (CNN) models, inspired
        by the Inception-v4 architecture, transformed mainly for Time Series
        Classification (TSC) tasks.

    Args:
        input_shape (tuple): The train data shape.
        include_top (bool): whether to include a fully-connected layer at
            the top of the network.
        num_classes (int): number of classes to predict. Default 1.
        classifier_activation (str or callable, optional): activation function
            (either as str or object) for the classification task.
        nb_filters (int): The number of nb filters.
        use_residual (bool): Whether to use a residual block or not.
        use_bottleneck (bool): Whether to use a bottleneck layer or not.
        depth (int): The depth of the network.
        kernel_size (int): The kernel size of the network.
        bottleneck_size (int): The number of output filters in the convolution.
        drp_input (float): Range 0-1.
        drp_high (float): Range 0-1.
        kernel_initialize (str): The variance scaling initializer.
            Default: "he_uniform".
        kernel_regularize (Union([float, None])): Regularizer to apply a
            penalty on the layer's kernel.Default: 4e-5.
        kernel_constraint (int): The constraint of the value of the incoming
            weights. Default 3.
        spatial (bool): Determines the type of Dropout. If True, it applies
            SpatialDropout1D else Monte Carlo Dropout. Default False.
        mc_inference (bool, optional):
            -If true, Dropout is enabled even during inference.
            -If False, Dropout is neither enabled on training nor
                during inference.
            -If None, Dropout is enabled during training but not during
                inference. Defaults to None.

    Returns:
        A Keras model instance.
    """

    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    kernel_size = kernel_size - 1

    # -- Initiating Model Topology --
    # define the input of the network
    input_layer = Input(shape=input_shape, name="input_layer")

    x = dropout_layer_1d(inputs=input_layer, drp_rate=drp_input,
                         spatial=spatial, mc_inference=mc_inference)

    x_incept = inception_block(inputs=x,
                               use_bottleneck=use_bottleneck,
                               bottleneck_size=bottleneck_size,
                               use_residual=use_residual,
                               activation=classifier_activation,
                               depth=depth,
                               nb_filters=nb_filters,
                               kernel_size=kernel_size,
                               kernel_initialize=kernel_initialize,
                               kernel_regularize=kernel_regularize,
                               kernel_constraint=kernel_constraint
                               )

    x = GlobalAveragePooling1D()(x_incept)

    # Dropout
    x = dropout_layer_1d(inputs=x, drp_rate=drp_high, spatial=spatial,
                         mc_inference=mc_inference)

    if include_top is True:
        # flatten
        x = Flatten()(x)
        # output
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x
    # model
    model = Model(inputs=input_layer, outputs=outputs)

    return model


def inception_block(inputs,
                    depth=6,
                    use_bottleneck=True,
                    bottleneck_size=1,
                    use_residual=True,
                    activation="softmax",
                    nb_filters=32,
                    kernel_size=41,
                    kernel_initialize="he_uniform",
                    kernel_regularize=None,
                    kernel_constraint=None):
    """
    Args:
        inputs:
        depth:
        use_bottleneck:
        bottleneck_size:
        use_residual:
        activation:
        nb_filters:
        kernel_size:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:
    """
    x = inputs
    inputs_res = inputs

    for d in range(depth):

        x = inception_module(inputs=x, use_bottleneck=use_bottleneck,
                             bottleneck_size=bottleneck_size,
                             activation=activation, nb_filters=nb_filters,
                             kernel_size=kernel_size,
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint)

        if use_residual and d % 3 == 2:
            residual_conv = Conv1D(filters=128,
                                   kernel_size=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_initializer=kernel_initialize,
                                   kernel_regularizer=kernel_regularize,
                                   kernel_constraint=kernel_constraint
                                   )(inputs_res)

            shortcut_y = BatchNormalization()(residual_conv)
            res_out = Add()([shortcut_y, x])
            x = Activation("relu")(res_out)
            inputs_res = x

    return x


def inception_module(inputs,
                     use_bottleneck=True,
                     bottleneck_size=32,
                     activation="softmax",
                     nb_filters=64,
                     kernel_size=41,
                     kernel_initialize="he_uniform",
                     kernel_regularize=None,
                     kernel_constraint=None,
                     stride=1):
    """
    Args:
        inputs:
        use_bottleneck:
        bottleneck_size:
        activation:
        nb_filters:
        kernel_size:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:
        stride:

    Returns:
    """
    if use_bottleneck and nb_filters > 1:
        x = Conv1D(filters=bottleneck_size, kernel_size=1,
                   padding="same", activation="linear", use_bias=False,
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(inputs)
    else:
        x = inputs

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
    print("kernel size list: ", kernel_size_s)

    conv_list = []
    for i in range(len(kernel_size_s)):
        print(f"Inception filters: {nb_filters} - kernel: {kernel_size_s[i]}")
        conv = Conv1D(filters=nb_filters,
                      kernel_size=kernel_size_s[i],
                      strides=stride,
                      padding="same",
                      activation=activation,
                      use_bias=False,
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint
                      )(x)

        conv_list.append(conv)

    x2 = MaxPooling1D(pool_size=3, strides=stride, padding="same")(inputs)

    # pass via a Conv1D to match the shapes
    last_conv = Conv1D(filters=nb_filters, kernel_size=1,
                       padding="same", activation=activation, use_bias=False,
                       kernel_initializer=kernel_initialize,
                       kernel_regularizer=kernel_regularize,
                       kernel_constraint=kernel_constraint
                       )(x2)

    conv_list.append(last_conv)

    x_concat = Concatenate(axis=2)(conv_list)

    x_bn = BatchNormalization()(x_concat)

    x_post = Activation("relu")(x_bn)

    return x_post
