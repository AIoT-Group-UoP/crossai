import numpy as np


def magnitude(*args):
    """Compute the magnitude of a signal using L2 norm

    Args:
        *args: Input signal. Can be a list of signals or a numpy array

    Returns:
        magnitude (float): Returns the magnitude of a signal
    """
    magnitude = 0
    for sig in args:
        if type(sig) is list:
            sig = np.array(sig)
        elif type(sig) is np.ndarray:
            pass
        else:
            raise ValueError("Please provide a list or a numpy array")

        magnitude += sig**2

    return np.sqrt(magnitude).astype(np.float32)
