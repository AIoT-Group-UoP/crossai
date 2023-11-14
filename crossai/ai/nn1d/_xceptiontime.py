import logging
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.layers import SeparableConv1D, Flatten, Add, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from .._layers_dropout import dropout_layer_1d
import tensorflow_addons as tfa


# Implementation of XceptionTime NN model based on:
# - https://arxiv.org/pdf/1911.03803.pdf
# - https://ieeexplore.ieee.org/document/9881698/
def XceptionTime(input_shape,
                 include_top=True,
                 num_classes=1,
                 classifier_activation='softmax',
                 xception_adaptive_size=50,
                 xception_adapt_ws_divide=4,
                 n_filters=16,
                 kernel_initialize="he_uniform",
                 kernel_regularize=1e-5,
                 kernel_constraint=3,
                 drp_input=0,
                 drp_mid=0,
                 drp_high=0,
                 spatial=False,
                 mc_inference=None):
    """A novel deep learning model referred to as the XceptionTime
    architecture. The proposed innovative XceptionTime is designed by the
    integration of depthwise separable convolutions, adaptive average pooling,
    and a novel non-linear normalization technique. By utilizing the depthwise
    separable convolutions, the XceptionTime network has far fewer parameters
    resulting in a less complex network. The updated architecture in this
    CrossAI topology is extended in such a way as to achieve higher confidence
    in the model’s predictions, it can be adapted to any window size, and its
    upgraded functionalities can avoid overfitting and achieve better model
    generalization.

    Args:
        input_shape (tuple): The shape of a single instance of the dataset.
        include_top (bool): whether to include a fully-connected layer at the
        top of the network.
        num_classes (int): number of classes to predict. Default 1.
        classifier_activation (tf.keras.activation): activation function
            (either as str or object) for the classification task.
        xception_adaptive_size (int): The adaptive size. Default: 50.
        xception_adapt_ws_divide (int): The number that will divide the
            adaptive size. Default 4.
        n_filters (int): The number of filters. Default: 16
        kernel_initialize (str): The variance scaling initializer.
            Default value: "he_uniform".
        kernel_regularize (float or str): Regularizer to apply a penalty on
            the layer's kernel. Can be float or str in 1e-5 format.
        kernel_constraint (int): The constraint of the value of the incoming
            weights. Default 3.
        drp_input (float): Fraction of the input units to drop in the input
            dropout layer. Default: 0.
        drp_mid (float): Fraction of the input units to drop in the
            mid-dropout layer. Default: 0.
        drp_high (float): Fraction of the input units to drop in the last
            dropout layer. Default: 0.
        spatial (Boolean): Determines the type of Dropout. If True, it applies
            SpatialDropout1D else Monte Carlo Dropout. Default: False.
        mc_inference (bool, optional):
            -If true, Dropout is enabled even during inference.
            -If False, Dropout is neither enabled on training nor
                during inference.
            -If None, Dropout is enabled during training but not during
                inference. Defaults to None.

    Returns:
        A Keras Model instance.
    """

    # confirm that things are working as expected
    logging.info(input_shape[0])
    logging.info(xception_adapt_ws_divide)

    # check and adjust adaptive size based on input shape
    if input_shape[0] % xception_adapt_ws_divide == 0:
        xception_adaptive_size = int(input_shape[0] / xception_adapt_ws_divide)
    else:
        xception_adaptive_size = xception_adaptive_size
        print("Provide a dividable number for the window size.")
        raise Exception("Provide a dividable number for the window size.")
    print(f"Input size W of window transformed into a fixed length of \
        {xception_adaptive_size} sample ""for AAP mid layer.")

    # regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # -- Initiating Model Topology --
    # input layer of the network
    input_layer = Input(shape=input_shape, name="input_layer")

    # Dropout
    x = dropout_layer_1d(inputs=input_layer, drp_rate=drp_input,
                         spatial=spatial, mc_inference=mc_inference)

    # COMPONENT 1 - Xception Block
    x = xception_block(inputs=x, n_filters=n_filters,
                       kernel_initialize=kernel_initialize,
                       kernel_regularize=kernel_regularize,
                       kernel_constraint=kernel_constraint)

    # COMPONENT 2
    # Head of the sequential component
    head_nf = n_filters * 32

    # transform the input with window size W to a fixed length
    # of adaptive size (default 50)
    x = tfa.layers.AdaptiveAveragePooling1D(xception_adaptive_size)(x)

    # Dropout
    x = dropout_layer_1d(inputs=x, drp_rate=drp_mid, spatial=spatial,
                         mc_inference=mc_inference)

    # stack 3 Conv1x1 Convolutions to reduce the time-series
    # to the number of the classes
    x = conv1d_block(x, nf=head_nf/2, drp_on=False, drp_rate=0.5,
                     spatial=True,
                     kernel_initialize=kernel_initialize,
                     kernel_regularize=kernel_regularize,
                     kernel_constraint=kernel_constraint)

    x = conv1d_block(x, nf=head_nf/4, drp_on=False, drp_rate=0.5,
                     spatial=True,
                     kernel_initialize=kernel_initialize,
                     kernel_regularize=kernel_regularize,
                     kernel_constraint=kernel_constraint)

    x = conv1d_block(x, nf=num_classes, drp_on=False, drp_rate=0.5,
                     spatial=True,
                     kernel_initialize=kernel_initialize,
                     kernel_regularize=kernel_regularize,
                     kernel_constraint=kernel_constraint)

    # convert the length of the input signal to 1 with the
    x = tfa.layers.AdaptiveAveragePooling1D(1)(x)

    # # Dropout
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


def conv1d_block(inputs, nf, ks=1, strd=1, pad="same", bias=False, bn=True,
                 act=True, act_func="relu", drp_on=False, drp_rate=0.5,
                 spatial=True, mc_inference=None, kernel_initialize=None,
                 kernel_regularize=None, kernel_constraint=None):

    """Create a block of layers consisting of Conv1D, BatchNormalization,
        Activation and Dropout.

    Args:
        inputs (tensor): Input tensor to the block.
        nf (int): Number of filters for the Conv1D layer.
        ks (int): Kernel size for the Conv1D layer. Defaults to 1.
        strd (int): Strides for the Conv1D layer. Defaults to 1.
        pad (str): Padding for the Conv1D layer. Defaults to "same".
        bias (bool): Whether to use bias in Conv1D. Defaults to False.
        bn (bool): Whether to use BatchNormalization. Defaults to True.
        act (bool): Whether to use an Activation layer. Defaults to True.
        act_func (str): Activation function to use. Defaults to "relu".
        drp_on (bool): Whether to include a Dropout layer. Defaults to False.
        drp_rate (float): Dropout rate. Defaults to 0.5.
        spatial (bool): Whether to use spatial dropout. Defaults to True.
        mc_inference (bool):
            -If true, Dropout is enabled even during inference.
            -If False, Dropout is neither enabled on training nor
                during inference.
            -If None, Dropout is enabled during training but not during
                inference. Defaults to None.
        kernel_initialize (str): Initialization for Conv1D layer.
            Defaults to None.
        kernel_regularize (str): Regularization for Conv1D layer.
            Defaults to None.
        kernel_constraint (str): Constraint for Conv1D layer.
            Defaults to None.

    Returns:
        tensor: Output tensor after applying the block of layers.
    """

    x = Conv1D(filters=int(nf),
               kernel_size=ks,
               strides=strd,
               padding=pad,
               use_bias=bias,
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint
               )(inputs)

    if bn:
        x = BatchNormalization()(x)

    if act:
        x = Activation(act_func)(x)

    if drp_on:
        x = dropout_layer_1d(x, drp_rate, spatial, mc_inference)

    return x


def xception_block(inputs, n_filters, depth=4, use_residual=True,
                   kernel_initialize=None, kernel_regularize=None,
                   kernel_constraint=None):

    """Xception Block function that applies a series of Xception modules,
        potentially with residual connections.

    Args:
        inputs (tensor): Input tensor to the block.
        n_filters (int): Number of filters for the convolutional layers.
        depth (int, optional): Depth of the Xception block. Default is 4.
        use_residual (bool, optional): Whether to use residual connections.
            Default is True.
        kernel_initialize (str, optional): Initializer for the kernel weights.
        kernel_regularize (str, optional): Regularizer for the kernel weights.
        kernel_constraint (str, optional): Constraint for the kernel weights.

    Returns:
        x (tensor): Output tensor after applying the Xception block.
    """

    x = inputs
    input_res = inputs

    for d in range(depth):
        xception_filters = n_filters * 2 ** d
        x = xception_module(x, xception_filters,
                            kernel_initialize=kernel_initialize,
                            kernel_regularize=kernel_regularize,
                            kernel_constraint=kernel_constraint)

        if use_residual and d % 2 == 1:
            residual_conv_filters = n_filters * 4 * (2 ** d)
            res_out = Conv1D(filters=residual_conv_filters, kernel_size=1,
                             padding="same", use_bias=False,
                             kernel_initializer=kernel_initialize,
                             kernel_regularizer=kernel_regularize,
                             kernel_constraint=kernel_constraint
                             )(input_res)

            shortcut_y = BatchNormalization()(res_out)
            res_out = Add()([shortcut_y, x])

            x = Activation("relu")(res_out)
            input_res = x

    return x


def xception_module(inputs, n_filters, use_bottleneck=True, kernel_size=41,
                    stride=1, kernel_initialize=None, kernel_regularize=None,
                    kernel_constraint=None):

    """XceptionModule function that applies a series of SeparableConv1D layers
        and a MaxPooling1D layer, followed by concatenation.

    Args:
        inputs (tensor): A Keras tensor serving as the starting point for this
            module.
        n_filters (int): Number of filters for the convolutional layers.
        use_bottleneck (bool): Whether to use a bottleneck Conv1D layer at the
            beginning. Default is True.
        kernel_size (int): Kernel size for the SeparableConv1D layers.
            Default is 41.
        stride (int): Stride for the SeparableConv1D and MaxPooling1D layers.
            Default is 1.
        kernel_initialize (str): Initializer for the kernel weights.
            Default is None.
        kernel_regularize (str): Regularizer for the kernel weights.
            Default is None.
        kernel_constraint (str): Constraint for the kernel weights. Default is
            None.

    Returns:
        x_post (tensor): Output tensor after applying the Xception module.
    """

    if use_bottleneck and n_filters > 1:
        x = Conv1D(filters=n_filters, kernel_size=1, padding="valid",
                   use_bias=False,
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(inputs)
    else:
        x = inputs

    # Assuming kernel_padding_size_lists function is defined
    kernel_sizes, padding_sizes = kernel_padding_size_lists(kernel_size)

    separable_conv_list = []
    for kernel in kernel_sizes:
        separable_conv = SeparableConv1D(filters=n_filters, kernel_size=kernel,
                                         strides=stride,
                                         padding="same", use_bias=False,
                                         kernel_initializer=kernel_initialize,
                                         kernel_regularizer=kernel_regularize,
                                         kernel_constraint=kernel_constraint
                                         )(x)
        separable_conv_list.append(separable_conv)

    x2 = MaxPooling1D(pool_size=3, strides=stride, padding="same")(inputs)
    x2 = Conv1D(filters=n_filters, kernel_size=1, padding="valid",
                use_bias=False,
                kernel_initializer=kernel_initialize,
                kernel_regularizer=kernel_regularize,
                kernel_constraint=kernel_constraint)(x2)

    separable_conv_list.append(x2)

    x_post = Concatenate(axis=2)(separable_conv_list)

    return x_post


def kernel_padding_size_lists(max_kernel_size):
    """
    Args:
        max_kernel_size (int):

    Returns:
    """
    i = 0
    kernel_size_list = []
    padding_list = []
    while i < 3:
        size = max_kernel_size // (2 ** i)
        if size == max_kernel_size:
            kernel_size_list.append(int(size))
            padding_list.append(int((size - 1) / 2))
        else:
            kernel_size_list.append(int(size + 1))
            padding_list.append(int(size / 2))
        i += 1

    return kernel_size_list, padding_list
