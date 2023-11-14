from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from crossai.ai import dropout_layer_1d


# Implementation of NN model for 1D based on:
# - https://ieeexplore.ieee.org/document/8488627
def BiLSTM_Time(input_shape,
                include_top=True,
                num_classes=1,
                n_layers=3,
                classifier_activation="softmax",
                lstm_units=[32, 32, 32],
                dense_units=[128],
                drp=0,
                spatial=False,
                mc_inference=None):
    """Constructs a deep neural network using bidirectional LSTM.

    This model converts low-level audio features into high-level expressions.

    Args:
        input_shape (tuple): Shape of the input data, excluding batch size.
        include_top (bool): If true, includes a fully-connected layer at the
            top.
        num_classes (int): Number of prediction classes. Defaults to 1.
        classifier_activation (str or callable, optional): Activation function
            for the classification task.
        n_layers (int): Number of Bidirectional LSTM layers. Defaults to 3.
        lstm_units (list of int): LSTM units for each layer. Defaults to
            [32, 32, 32].
        Dense_units (list of int): Units for each dense layer. Defaults to
            [128].
        drp (float): Dropout rate. Defaults to 0.
        spatial (bool): Type of Dropout. True for SpatialDropout1D, False for
            Monte Carlo Dropout. Defaults to False.
        mc_inference (bool, optional): Dropout setting during inference. True
            enables Dropout, False disables it, None for training only.
                Defaults to None.

    Returns:
        keras.Model: A Keras model instance.
    """

    input_layer = Input(shape=input_shape, name="input_layer")

    x = Bidirectional(LSTM(units=lstm_units[0], activation="tanh",
                           return_sequences=True))(input_layer)
    x = dropout_layer_1d(x, drp, spatial=spatial, mc_inference=mc_inference)

    x_block = bilstm_block(x, n_layers, lstm_units, drp)
    x_dense = dense_block(x_block, n_layers, dense_units)

    if include_top is True:
        outputs = Dense(num_classes, activation=classifier_activation)(x_dense)
    else:
        outputs = x_dense

    model = Model(inputs=input_layer, outputs=outputs)

    return model


def bilstm_block(inputs, n_layers, lstm_units, drp_rate, mc_inference=None):
    """
    Constructs a bidirectional LSTM (BiLSTM) block.

    Args:
        inputs: Input tensor for the BiLSTM block.
        n_layers (int): Number of LSTM layers in the block.
        lstm_units (list of int): Number of units in each LSTM layer.
        drp_rate (float): Dropout rate to be applied after each LSTM layer.
        mc_inference (bool, optional): If True, enables Monte Carlo dropout
            during inference. Defaults to None.

    Returns:
        The output tensor from the last layer of the BiLSTM block.
    """

    x = inputs
    for i in range(1, n_layers - 1):
        x = Bidirectional(LSTM(units=lstm_units[i], return_sequences=True,
                               activation="tanh"))(x)
        x = dropout_layer_1d(x, drp_rate, False, mc_inference)

    x = Bidirectional(LSTM(units=lstm_units[-1], return_sequences=False,
                           activation="tanh"))(x)
    x = dropout_layer_1d(x, drp_rate, False, mc_inference)

    return x


def dense_block(inputs, n_layers, dense_units):
    """
    Builds a block of dense layers.

    Args:
        inputs: Input tensor for the dense layers.
        n_layers (int): Number of dense layers to be created.
        dense_units (list of int): Number of units in each dense layer.

    Returns:
        The output tensor from the last dense layer.
    """

    x = inputs
    for d in range(0, min(n_layers, len(dense_units))):
        x = Dense(units=dense_units[d], activation="relu")(x)

    return x
