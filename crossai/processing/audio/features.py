import numpy as np


def loudness(sig):
    """Calculates the loudness of a audio signal.

    Args:
        x (np.ndarray): Input signal.

    Returns:
        float: Loudness
    """
    rms = np.sqrt(np.mean(sig**2))
    loudness = 20 * np.log10(rms)  # to db
    return loudness
