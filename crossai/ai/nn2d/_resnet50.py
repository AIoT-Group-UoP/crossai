from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import Model
from .._layers_dropout import dense_drop_block


# This architecture is based on ResNet 50 (2015)
# Paper: https://arxiv.org/pdf/1512.03385.pdf
def ResNet50(input_shape,
             include_top=True,
             num_classes=1,
             classifier_activation="softmax",
             kernel_initialize="he_normal",
             kernel_regularize=1e-3,
             kernel_constraint=3,
             dense_layers=0,
             dense_units=[128, 128],
             dropout=False,
             dropout_first=False,
             dropout_rate=[0.5, 0.5],
             spatial=False,
             mc_inference=None
             ):
    """ResNet50 Model

    Args:
        input_shape (tuple)): The shape of a single instance of the dataset.
        include_top (bool, optional): whether to include a fully-connected
            layer at the top of the network.
        num_classes (int, optional): number of classes to predict. Default 1.
        classifier_activation (Union[str, Callable], optional): activation
            function (either as str or object) for the classification task.
        kernel_initialize (str, optional): The variance scaling initializer.
            Default: "he_uniform".
        kernel_regularize (Union[str, float], optional): Regularizer to apply
            penalty on the layer"s kernel. Can be float or str in 1e-5 format.
        kernel_constraint (int, optional): The constraint of the value of the
            incoming weights. Default 3.
        dense_layers (int, optional): Number of dense layers. Default 0.
        dense_units (List[int], optional): Number of units per dense layer.
            Default [128, 128]
        dropout (bool, optional): whether to use dropout or not. Default False.
        dropout_first (bool, optional): Add dropout before dense layer or
            after. Default False.
        dropout_rate (List[float]): dropout rate for each dropout layer.
            Default 0.5.
        spatial (bool, optional): Determines the type of Dropout. If True, it
            applies SpatialDropout2D else
        Monte Carlo Dropout. Default: False.
        mc_inference (bool, optional):
        - If true, Dropout is enabled even during inference.
        - If False, Dropout is neither enabled on training nor during
            inference.
        - If None, Dropout is enabled during training but not during inference.
            Defaults to None.

    Returns:
        A Keras Model instance.
    """

    # Regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # Begin model topology
    input_layer = Input(shape=input_shape, name="input_layer")

    # Initial convolution block
    x = conv_bn_relu(inputs=input_layer, n_filters=64, kernel_size=7,
                     strides=2, kernel_initialize=kernel_initialize,
                     kernel_regularize=kernel_regularize,
                     kernel_constraint=kernel_constraint)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # Add ResNet blocks
    for filters, reps, s in zip([64, 128, 256, 512],
                                [3, 4, 6, 3],
                                [1, 2, 2, 2]):
        x = resnet_block(x,
                         filters,
                         reps,
                         s,
                         kernel_initialize,
                         kernel_regularize,
                         kernel_constraint)

    # Add top layer if specified
    if include_top:
        x = GlobalAvgPool2D()(x)
        x = Flatten()(x)
        x = dense_drop_block(inputs=x, n_layers=dense_layers,
                             dense_units=dense_units,
                             dropout=dropout, drop_first=dropout_first,
                             drop_rate=dropout_rate,
                             activation_dense="relu",
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint,
                             spatial=spatial,
                             mc_inference=mc_inference)
        outputs = Dense(units=num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    # Construct and return the model
    model = Model(input_layer, outputs)
    return model


def resnet_block(x, n_filters, reps, strides, kernel_initialize,
                 kernel_regularize, kernel_constraint):
    """
    Create a ResNet block with multiple residual layers.

    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of filters.
        reps (int): Number of repetitions.
        strides (int): Strides for convolution.
        kernel_initialize (str): Initialization method for the layers.
        kernel_regularize (float): Regularization factor.
        kernel_constraint (int): Max norm constraint for kernel values.

    Returns:
        tensor: Output tensor.
    """
    x = projection_block(x, n_filters, strides, kernel_initialize,
                         kernel_regularize, kernel_constraint)
    for _ in range(reps-1):
        x = identity_block(x, n_filters, kernel_initialize, kernel_regularize,
                           kernel_constraint)
    return x


def projection_block(x, n_filters, strides, kernel_initialize,
                     kernel_regularize, kernel_constraint):
    """
    Create a ResNet block with multiple residual layers.

    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of filters.
        strides (int): Strides for convolution.
        kernel_initialize (str): Initialization method for the layers.
        kernel_regularize (float): Regularization factor.
        kernel_constraint (int): Max norm constraint for kernel values.

    Returns:
        tensor: Output tensor.
    """
    shortcut = _conv_bn(x, 4*n_filters, 1, strides, kernel_initialize,
                        kernel_regularize, kernel_constraint)
    x = conv_bn_relu(x, n_filters, 1, strides, kernel_initialize,
                     kernel_regularize, kernel_constraint)
    x = conv_bn_relu(x, n_filters, 3, 1, kernel_initialize, kernel_regularize,
                     kernel_constraint)
    x = _conv_bn(x, 4*n_filters, 1, 1, kernel_initialize, kernel_regularize,
                 kernel_constraint)
    x = Add()([shortcut, x])
    return ReLU()(x)


def identity_block(x, n_filters, kernel_initialize, kernel_regularize,
                   kernel_constraint):
    """
    Create a ResNet block with multiple residual layers.

    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of filters.
        strides (int): Strides for convolution.
        kernel_initialize (str): Initialization method for the layers.
        kernel_regularize (float): Regularization factor.
        kernel_constraint (int): Max norm constraint for kernel values.

    Returns:
        tensor: Output tensor.
    """
    shortcut = x
    x = conv_bn_relu(x, n_filters, 1, 1, kernel_initialize, kernel_regularize,
                     kernel_constraint)
    x = conv_bn_relu(x, n_filters, 3, 1, kernel_initialize, kernel_regularize,
                     kernel_constraint)
    x = _conv_bn(x, 4*n_filters, 1, 1, kernel_initialize, kernel_regularize,
                 kernel_constraint)
    x = Add()([shortcut, x])
    return ReLU()(x)


def conv_bn_relu(inputs, n_filters,
                 kernel_size, strides,
                 kernel_initialize,
                 kernel_regularize,
                 kernel_constraint):
    """
    Create a ResNet block with multiple residual layers.

    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of filters.
        strides (int): Strides for convolution.
        kernel_initialize (str): Initialization method for the layers.
        kernel_regularize (float): Regularization factor.
        kernel_constraint (int): Max norm constraint for kernel values.

    Returns:
        tensor: Output tensor.
    """
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
               padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(inputs)
    x = BatchNormalization()(x)
    return ReLU()(x)


def _conv_bn(inputs, n_filters,
             kernel_size, strides,
             kernel_initialize,
             kernel_regularize,
             kernel_constraint):
    """
    Create a ResNet block with multiple residual layers.

    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of filters.
        strides (int): Strides for convolution.
        kernel_initialize (str): Initialization method for the layers.
        kernel_regularize (float): Regularization factor.
        kernel_constraint (int): Max norm constraint for kernel values.

    Returns:
        tensor: Output tensor.
    """
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(inputs)
    return BatchNormalization()(x)
