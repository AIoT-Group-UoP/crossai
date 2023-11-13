from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D
from .._layers_dropout import dropout_layer_1d


def CNN1D(input_shape,
          include_top=True,
          num_classes=1,
          classifier_activation="softmax",
          drp=0.,
          spatial=False,
          mc_inference=None):
    """
    Creates a simple 1D CNN model for experimental purposes.

    Args:
        input_shape (tuple): Shape of the input data, excluding the batch size.
        include_top (bool, optional): If true, includes a fully-connected layer
            at the top of the model. Defaults to True.
        num_classes (int, optional): Number of classes for prediction.
            Defaults to 1.
        classifier_activation (str or callable, optional): Activation function
            for the classification layer.
        drp (float, optional): Dropout rate. Defaults to 0.
        sptial (bool, optional): If true, applies Spatial Dropout, else applies
            standard Dropout. Defaults to False.
        mc_inference (bool, optional): If true, enables Dropout during
            inference. Defaults to False.

    Returns:
        keras.Model: An instance of a Keras Model.
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

    x = dropout_layer_1d(inputs=x, drp_rate=drp, spatial=spatial,
                         mc_inference=mc_inference)
    if include_top is True:
        x = Flatten()(x)
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(inputs=input_layer, outputs=outputs)

    return model
