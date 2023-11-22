import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D
from crossai.ai import dropout_layer_1d


def CNN1D(
    input_shape: tuple,
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: str = "softmax",
    drp_rate: float = 0.,
    spatial: bool = False,
    mc_inference: bool = None
) -> tf.keras.Model:
    """Creates a simple 1D CNN model for experimental purposes.

    Args:
        input_shape: Shape of the input data, excluding the batch size.
        include_top: If true, includes a fully-connected layer at the top of
            the model. Set to False for a custom layer.
        num_classes: Number of classes for prediction.
        classifier_activation: Activation function for the classification task.
        drp_rate: Dropout rate.
        spatial: If true, applies Spatial Dropout, else applies standard
            Dropout.
        mc_inference: If True, enables Monte Carlo dropout
            during inference. Defaults to None.

    Returns:
        model: An instance of a Keras Model.
    """

    # Define input tensor for the network, batch size is omitted
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Conv1D(64, 3, activation="relu")(input_layer)
    x = Conv1D(64, 3, activation="relu")(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation="relu")(x)
    x = Conv1D(128, 3, activation="relu")(x)

    # retain tensor shape (keepdims) since Spatial Dropout expects 3D input
    x = GlobalAveragePooling1D(keepdims=True if spatial else False)(x)

    x = dropout_layer_1d(inputs=x, drp_rate=drp_rate, spatial=spatial,
                         mc_inference=mc_inference)
    if include_top is True:
        x = Flatten()(x)
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(inputs=input_layer, outputs=outputs)

    return model
