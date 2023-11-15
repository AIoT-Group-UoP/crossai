import numpy as np
import scipy
from scipy import signal
import librosa


def spectral_skewness(sig):
    """Computes the spectral skewness of a spectrogram or a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        spec_skewness (float): Returns the spectral skewness of a spectrogram
                                or a signal
    """

    spec_skewness = scipy.stats.skew(sig)

    return spec_skewness


def max(sig):
    """Computes the maximum value of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        max_value (float): Returns the maximum value of a signal
    """

    return np.max(sig)


def min(sig):
    """Computes the minimum value of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        min_value (float): Returns the minimum value of a signal
    """

    return np.min(sig)


def mean(sig):
    """Computes the mean value of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        mean_value (float): Returns the mean value of a signal
    """

    return np.mean(sig)


def std(sig):
    """Computes the standard deviation of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        std_value (float): Returns the standard deviation of a signal
    """

    return np.std(sig)


def kurtosis(sig):
    """Computes the kurtosis of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        kurtosis (float): Returns the kurtosis of a signal
    """

    return scipy.stats.kurtosis(sig)


def energy(sig):
    """Computes the energy of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        energy (float): Returns the energy of a signal
    """

    return np.sum(sig**2)/len(sig)


def spectral_entropy(sig):
    """Computes the spectral entropy of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        spectral_entropy (float): Returns the spectral entropy of a signal
    """

    return scipy.stats.entropy(sig)


def flux(sig):
    """Computes the flux of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        flux (float): Returns the flux of a signal
    """

    return librosa.onset.onset_strength(sig)


def rms_value(sig):
    """Computes the Root Mean Square (RMS) of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        rms_value (float): Returns the root mean square of a signal
    """

    return librosa.feature.rms(sig).flatten()


def spectral_centroid(sig):
    """Computes the spectral centroid of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        spectral_centroid (float): Returns the spectral centroid of a signal
    """

    return librosa.feature.spectral_centroid(sig).flatten()


def envelope(sig):
    """Computes the envelope of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        envelope (numpy array): Returns the envelope of a signal
    """

    return np.abs(signal.hilbert(sig))
