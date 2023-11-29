from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from .._layers_dropout import dense_drop_block


# Implementation of Xception NN model based on:
# - https://arxiv.org/abs/1409.1556
def VGG16(input_shape,
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
    """ VGG16 Model

    Args:
        input_shape (tuple)): The shape of a single instance of the dataset.
        include_top (bool, optional): whether to include a fully-connected
            layer at the top of the network.
        num_classes (int, optional): number of classes to predict. Default 1.
        classifier_activation (Union[str, Callable], optional): activation
            function (either as str or object) for the classification task.
        kernel_initialize (str, optional): The variance scaling initializer.
            Default: "he_uniform".
        kernel_regularize (Union[str, float], optional): A penalty on the layer
            kernel. Can be float or a str in scientific notation (e.g. '1e-5').
            Default: 1e-5.
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
            applies SpatialDropout2D else Monte Carlo Dropout. Default: False.
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

    # Blocks of VGG16
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(input_layer)
    x = ReLU()(x)

    # 1 conv layer, 64 channels
    x = vgg_block(x, 1, 64, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 2 conv layers, 128 channels
    x = vgg_block(x, 2, 128, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 3 conv layers, 256 channels
    x = vgg_block(x, 3, 256, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 3 conv layers, 512 channels
    x = vgg_block(x, 3, 512, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 3 conv layers, 512 channels
    x = vgg_block(x, 3, 512, kernel_initialize, kernel_regularize,
                  kernel_constraint)

    if include_top:
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

        outputs = Dense(units=num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(input_layer, outputs, name="VGG_16")
    return model


def vgg_block(input_tensor, num_convs, num_channels,
              kernel_initialize, kernel_regularize, kernel_constraint):
    """
    Adds a VGG block to the model.

    Args:
        input_tensor (keras Tensor): Input tensor for the block.
        num_convs (int): Number of convolutional layers in the block.
        num_channels (int): Number of filters/channels for the convolutional
            layer.
        kernel_initialize (str): The variance scaling initializer.
        kernel_regularize (Union[str, float], optional): A penalty on the layer
            kernel. Can be float or a str in scientific notation (e.g. '1e-5').
        kernel_constraint (int): The constraint of the value of the incoming
            weights.

    Returns:
        keras Tensor: Tensor after passing through the block.
    """
    x = input_tensor
    for _ in range(num_convs):
        x = Conv2D(filters=num_channels, kernel_size=(3, 3), padding="same",
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint)(x)
        x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x
