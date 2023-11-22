from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, ReLU
from tensorflow.keras.layers import Add, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from .._layers_dropout import dense_drop_block


# This architecture is based on ResNet 34 (2015)
# Paper: https://arxiv.org/pdf/1512.03385.pdf
def ResNet34(input_shape,
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
    """ResNet34 Model

    Args:
        input_shape (tuple): The shape of a single instance of
            the dataset.
        include_top (bool, optional): whether to include a fully-connected
            layer at the top of the network.
        num_classes (int, optional): number of classes to predict. Default 1.
        classifier_activation (Union[str, callable], optional): activation
            function for the classification task.
        kernel_initialize (str, optional): The variance scaling initializer.
            Default value: "he_uniform".
        kernel_regularize (Union[float, str], optional): Regularizer to apply a
            penalty on the layer"s kernel. Can be float or str in 1e-5 format.
        kernel_constraint (int, optional): The constraint of the value of the
            incoming weights. Default 3.
        dense_layers (int, optional): Number of dense layers. Default 0.
        dense_units (List[int], optional): Number of units per dense layer.
            Default [128, 128]
        dropout (bool, optional): whether to use dropout or not. Default False.
        dropout_first (bool, optional): Add dropout before dense layer or
            after. Default False.
        dropout_rate (List[float], optional): dropout rate for each dropout
            layer. Default 0.5.
        spatial (bool, optional): Determines the type of Dropout. If True, it
            applies SpatialDropout2D else Monte Carlo Dropout. Default: False.
        mc_inference (bool, optional):
            -If true, Dropout is enabled even during inference.
            -If False, Dropout is neither enabled on training nor during
                inference.
            -If None, Dropout is enabled during training but not during
                inference. Defaults to None.

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
    # input layer
    input_layer = Input(shape=input_shape, name="input_layer")
    # x = ZeroPadding2D((3, 3))(input_layer)

    # The Stem Convolution Group
    x = stem(inputs=input_layer,
             kernel_initialize=kernel_initialize,
             kernel_regularize=kernel_regularize,
             kernel_constraint=kernel_constraint
             )

    # The learner
    x = learner(x=x,
                kernel_initialize=kernel_initialize,
                kernel_regularize=kernel_regularize,
                kernel_constraint=kernel_constraint
                )

    if include_top:
        # flatten
        x = Flatten()(x)

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
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(input_layer, outputs, name="Resnet_34")
    return model


def stem(inputs,
         kernel_initialize="he_normal",
         kernel_regularize=1e-3,
         kernel_constraint=3
         ):
    """ Construct the Stem Convolution Group
        inputs:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:
    """
    # First Convolutional layer, where pooled
    # feature maps will be reduced by 75%
    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint
               )(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    return x


def learner(x,
            kernel_initialize="he_normal",
            kernel_regularize=1e-3,
            kernel_constraint=3
            ):
    """ Construct the Learner
        x:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:
    """
    # First Residual Block Group of 64 filters
    x = residual_group(x, 64, 3, True, kernel_initialize, kernel_regularize,
                       kernel_constraint)

    # Second Residual Block Group of 128 filters
    x = residual_group(x, 128, 3, True, kernel_initialize, kernel_regularize,
                       kernel_constraint)

    # Third Residual Block Group of 256 filters
    x = residual_group(x, 256, 5, True, kernel_initialize, kernel_regularize,
                       kernel_constraint)

    # Fourth Residual Block Group of 512 filters
    x = residual_group(x, 512, 2, False, kernel_initialize, kernel_regularize,
                       kernel_constraint)
    return x


def residual_group(x,
                   n_filters,
                   n_blocks,
                   conv=True,
                   kernel_initialize="he_normal",
                   kernel_regularize=1e-3,
                   kernel_constraint=3
                   ):
    """ Construct a Residual Group
        x                 :
        n_filters         : number of filters
        n_blocks          : number of blocks in the group
        conv              : flag to include the convolution block connector
        kernel_initialize :
        kernel_regularize :
        kernel_constraint :
    """
    for _ in range(n_blocks):
        x = residual_block(x, n_filters, kernel_initialize, kernel_regularize,
                           kernel_constraint)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2)
    # to fit the next Residual Group
    if conv:
        x = conv_block(x, n_filters * 2, kernel_initialize, kernel_regularize,
                       kernel_constraint)
    return x


def residual_block(x, n_filters, kernel_initialize, kernel_regularize,
                   kernel_constraint):
    """ Construct a Residual Block of Convolutions
        x                 :
        n_filters         :
        kernel_initialize :
        kernel_regularize :
        kernel_constraint :
    """
    # skip connection
    shortcut = x

    # First Layer
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(x)
    x = BatchNormalization()(x)  # axis = -1 = 3
    x = ReLU()(x)

    # Second Layer
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(x)
    x = BatchNormalization()(x)

    # Add residue
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x


def conv_block(x, n_filters, kernel_initialize, kernel_regularize,
               kernel_constraint):
    """ Construct Block of Convolutions without Pooling
        x        : input into the block
        n_filters: number of filters
    """
    x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x
