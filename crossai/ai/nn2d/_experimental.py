from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten
from tensorflow.keras import Model
from crossai.ai import dropout_layer_1d, dropout_layer_2d


def CNN2D(input_shape,
          include_top=True,
          num_classes=1,
          classifier_activation="softmax",
          drp=0.,
          spatial=False,
          mc_inference=None):
    """
    Creates a simple 2D CNN model for experimental purposes.

    Args:
        input_shape (tuple): Shape of the input data, excluding the batch size.
        include_top (bool, optional): If true, includes a fully-connected layer
            at the top of the model. Defaults to True.
        num_classes (int, optional): Number of classes for prediction.
            Defaults to 1.
        classifier_activation (Union[str, Callable], optional): Activation
            function for the classification layer.
        drp (float, optional): Dropout rate. Defaults to 0.
        spatial (bool, optional): If true, applies Spatial Dropout, else
            applies standard Dropout. Defaults to False.
        mc_inference (Union[bool, None], optional): If true, enables Dropout
            during inference; None for inference without Dropout.
            Defaults to False.

    Returns:
        keras.Model: An instance of a Keras Model.
    """

    # Define input tensor for the network, batch size is omitted
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Conv2D(32, (3, 3), activation="relu")(input_layer)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(3)(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)

    if include_top is True:
        # retain tensor shape (keepdims) since Spatial Dropout expects 4D input
        x = GlobalAveragePooling2D(keepdims=True)(x)
        x = dropout_layer_2d(inputs=x, drp_rate=drp, spatial=spatial,
                             mc_inference=mc_inference)
        x = Flatten()(x)

        x = Dense(128, activation="relu")(x)
        x = dropout_layer_1d(inputs=x, drp_rate=drp)

        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(inputs=input_layer, outputs=outputs)

    return model
