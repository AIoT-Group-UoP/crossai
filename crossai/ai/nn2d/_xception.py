from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from .._layers_dropout import dense_drop_block


# Implementation of Xception NN model based on:
# - https://arxiv.org/abs/1610.02357
def Xception(input_shape,
             include_top=True,
             num_classes=1,
             classifier_activation="softmax",
             kernel_initialize="he_uniform",
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
    """Xception Model.

    Args:
        input_shape (tuple): The shape of a single instance of the dataset.
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

    # regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # -- Initiating Model Topology --
    # Create the input vector
    input_layer = Input(shape=input_shape, name="input_layer")

    # Create entry section
    x = entry_flow(inputs=input_layer,
                   kernel_initialize=kernel_initialize,
                   kernel_regularize=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )

    # Create the middle section
    x = middle_flow(x=x,
                    kernel_initialize=kernel_initialize,
                    kernel_regularize=kernel_regularize,
                    kernel_constraint=kernel_constraint
                    )

    # Create the exit section for 2 classes
    x = exit_flow(x=x,
                  kernel_initialize=kernel_initialize,
                  kernel_regularize=kernel_regularize,
                  kernel_constraint=kernel_constraint
                  )

    if include_top:
        # flatten
        x = Flatten()(x)

        # apply multiple sequential dense/dropout layers
        x = dense_drop_block(inputs=x, n_layers=dense_layers,
                             dense_units=dense_units,
                             dropout=dropout, drop_first=dropout_first,
                             drop_rate=dropout_rate,
                             activation_dense="relu",
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint,
                             spatial=spatial,
                             mc_inference=mc_inference
                             )

        # Fully connected output layer (classification)
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(input_layer, outputs, name="Xception")

    return model


def entry_flow(inputs,
               kernel_initialize,
               kernel_regularize,
               kernel_constraint):
    """Creates the entry flow section.

    Args:
        inputs: Input tensor to neural network.
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """

    def stem(inputs, kernel_initialize, kernel_regularize, kernel_constraint):
        """Creates the stem entry into the neural network.

        Args:
            inputs: Input tensor to neural network.
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:

        Returns:

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
        x = projection_block(x, n_filters, kernel_initialize,
                             kernel_regularize, kernel_constraint)

    return x


def middle_flow(x, kernel_initialize, kernel_regularize, kernel_constraint):
    """Creates the middle flow section

    Args:
        x: Input tensor into section.
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    # Create 8 residual blocks
    for _ in range(8):
        x = residual_block(x, 728, kernel_initialize, kernel_regularize,
                           kernel_constraint)
    return x


def exit_flow(x, kernel_initialize, kernel_regularize, kernel_constraint):
    """Creates the exit flow section.

    Args:
        x: Input to the exit flow section.
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:
        Output tensor of exit flow section.
    """

    # 1x1 strided convolution to increase number and reduce size of
    # feature maps in identity link to match output of residual block for
    # the add operation (projection shortcut)
    shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding="same",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint
                      )(x)
    shortcut = BatchNormalization()(shortcut)

    x = ReLU()(x)
    # First Depthwise Separable Convolution
    # Dimensionality reduction - reduce number of filters
    x = SeparableConv2D(728, (3, 3), padding="same",
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
    x = SeparableConv2D(1024, (3, 3), padding="same",
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Add the projection shortcut to the output of the pooling layer
    x = Add()([x, shortcut])

    # Third Depthwise Separable Convolution
    x = SeparableConv2D(1556, (3, 3), padding="same",
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
    x = SeparableConv2D(2048, (3, 3), padding="same",
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Global Average Pooling will flatten the 10x10 feature maps into 1D
    # feature maps
    output = GlobalAveragePooling2D()(x)

    return output


def projection_block(x, n_filters, kernel_initialize, kernel_regularize,
                     kernel_constraint):
    """Creates a residual block using Depth-wise Separable Convolutions
    with Projection shortcut.

    Args:
        x: Input tensor into residual block.
        n_filters: An integer that indicates the number of filters.
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    # Remember the input
    shortcut = x

    # Strided convolution to double number of filters in identity link to
    # match output of residual block for the add operation
    # (projection shortcut)
    shortcut = Conv2D(n_filters, (1, 1), strides=(2, 2), padding="same",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint
                      )(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # ReLu activation is applied before SeparableConv2D
    # only in the last two projections
    if n_filters in [256, 728]:
        x = ReLU()(x)

    # First Depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding="same",
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
    x = SeparableConv2D(n_filters, (3, 3), padding="same",
                        depthwise_initializer=kernel_initialize,
                        pointwise_initializer=kernel_initialize,
                        depthwise_regularizer=kernel_regularize,
                        pointwise_regularizer=kernel_regularize,
                        depthwise_constraint=kernel_constraint,
                        pointwise_constraint=kernel_constraint
                        )(x)
    x = BatchNormalization()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Add the projection shortcut to the output of the block
    x = Add()([x, shortcut])

    return x


def residual_block(x, n_filters, kernel_initialize, kernel_regularize,
                   kernel_constraint):
    """Creates a residual block using Depth-wise Separable Convolutions.

    Args:
        x: Input into residual block.
        n_filters: An integer that indicates the number of filters.
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    # Remember the input
    shortcut = x

    for _ in range(3):
        x = ReLU()(x)
        # First Depthwise Separable Convolution
        x = SeparableConv2D(n_filters, (3, 3), padding="same",
                            depthwise_initializer=kernel_initialize,
                            pointwise_initializer=kernel_initialize,
                            depthwise_regularizer=kernel_regularize,
                            pointwise_regularizer=kernel_regularize,
                            depthwise_constraint=kernel_constraint,
                            pointwise_constraint=kernel_constraint
                            )(x)
        x = BatchNormalization()(x)

    # Add the identity link to the output of the block
    x = Add()([x, shortcut])
    return x
