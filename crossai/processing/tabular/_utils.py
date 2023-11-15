import warnings
import numpy as np
warnings.filterwarnings('ignore')


def axis_to_model_shape(*args):
    """Function to convert one instance of multiple axes data to model shape
    (window_size,features).

    Args:
        *args: Each axis data/ feature.

    Returns:
        data (numpy array): Data in model shape.

    """
    data = np.array(args[0])
    data = np.expand_dims(data, axis=0)
    data = data.T
    for feature in args[1:]:
        # concatenate so that the data will be in the required shape
        feature = np.expand_dims(feature, axis=0)
        data = np.concatenate((data, feature.T), axis=1)
    data = np.array(data.tolist())
    return data
