from tensorflow import keras
from tensorflow.keras.layers import Dropout, SpatialDropout1D, SpatialDropout2D
from tensorflow.keras.layers import Dense


@keras.saving.register_keras_serializable(package="MyLayers")
class MCDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)


@keras.saving.register_keras_serializable(package="MyLayers")
class MCSpatialDropout1D(SpatialDropout1D):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def call(self, inputs, training):
        return super().call(inputs, training=training)


@keras.saving.register_keras_serializable(package="MyLayers")
class MCSpatialDropout2D(SpatialDropout2D):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def call(self, inputs, training):
        return super().call(inputs, training=training)


def dropout_layer_1d(inputs, drp_rate=0.1, spatial=False, mc_inference=None):
    """Creates a Dropout layer for a 1D model.

    Args:
        inputs (Tensor): The input tensor.
        drp_rate (float, optional): Float between 0 and 1. Fraction of the
            input units to drop. Defaults to 0.1.
        spatial (bool, optional): If true, a Spatial 1D version of Dropout is
            applied. It drops entire 1D feature maps instead of individual
            elements. Defaults to False.
        mc_inference (bool, optional):
            -If true, Dropout is enabled even during inference.
            -If False, Dropout is neither enabled on training nor during
                inference.
            -If None, Dropout is enabled during training but not during
                inference. Defaults to None.

    Returns:
        Tensor: Output tensor after applying dropout.
    """
    if spatial:
        drp = MCSpatialDropout1D(drp_rate)(inputs, mc_inference)
    else:
        drp = MCDropout(drp_rate)(inputs, mc_inference)
    return drp


def dropout_layer_2d(inputs, drp_rate=0.1, spatial=False, mc_inference=None):
    """Creates a Dropout layer for a 2D model.

    Args:
        inputs (Tensor): The input tensor.
        drp_rate (float, optional): Float between 0 and 1. Fraction of the
            input units to drop. Defaults to 0.1.
        spatial (bool, optional): If true, a Spatial 1D version of Dropout is
            applied. It drops entire 1D feature maps instead of individual
            elements. Defaults to False.
        mc_inference (bool, optional):
            -If true, Dropout is enabled even during inference.
            -If False, Dropout is neither enabled on training nor during
                inference.
            -If None, Dropout is enabled during training but not during
                inference. Defaults to None.

    Returns:
        Tensor: Output tensor after applying dropout.
    """
    if spatial:
        drp = MCSpatialDropout2D(drp_rate)(inputs, mc_inference)
    else:
        drp = MCDropout(drp_rate)(inputs, mc_inference)
    return drp


def dense_drop_block(inputs, n_layers, dense_units, dropout, drop_rate,
                     drop_first=False, activation_dense="relu",
                     kernel_initialize=None, kernel_regularize=None,
                     kernel_constraint=None, spatial=False, mc_inference=None):
    """A layer block that can initialize a series of Dropout/Dense,
    Dense/Dropout, or Dense layers.

    Args:
        inputs (tensor): Input tensor
        n_layers (int): Number of dense-dropout pairs or dense layers
        dense_units: List of integers, number of dense units at each layer
        dropout (bool): Bool value whether to use dropout or not
        drop_rate (List[float]): dropout rate for each dropout layer
        drop_first (bool, optional): Boolean, whether to add dropout before
            dense layer or after
        activation_dense (Union[str, Callable], optional): Activation function
            to use in dense layers
        kernel_initialize (str, optional): Kernel initializer for dense layers
        kernel_regularize (str, optional): Kernel regularizer for dense layers
        kernel_constraint (str, optional): Kernel constraint for dense layers
        mc_dropout (bool, optional): Boolean, whether to use Monte Carlo Drp
        spatial (bool, optional): Boolean, whether to use Spatial Dropout
        mc_inference (bool):
            -If true, Dropout is enabled even during inference.
            -If False, Dropout is neither enabled on training nor during
                inference.
            -If None, Dropout is enabled during training but not during
                inference. Defaults to None.

    Returns:
        Output tensor after applying the dense and dropout layers
    """
    x = inputs
    for d in range(0, n_layers):
        if dropout and drop_first:
            x = dropout_layer_1d(inputs=x,
                                 drp_rate=drop_rate[d],
                                 spatial=spatial,
                                 mc_inference=mc_inference)

        x = Dense(units=dense_units[d],
                  kernel_initializer=kernel_initialize,
                  kernel_regularizer=kernel_regularize,
                  kernel_constraint=kernel_constraint,
                  activation=activation_dense)(x)

        if dropout and not drop_first:
            x = dropout_layer_1d(x, drp_rate=drop_rate[d], spatial=spatial,
                                 mc_inference=mc_inference)

    return x
