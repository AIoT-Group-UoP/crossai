import numpy as np
import scipy
from scipy import signal
import librosa


def spectral_skewness(sig: np.ndarray) -> float:
    """Computes the spectral skewness of a spectrogram or a signal.

    Calculates the skewness of a given spectrogram or signal.
    The skewness is a measure of the asymmetry of the probability
    distribution of a real-valued random variable about its mean. 

    Args:
        sig: The input signal or spectrogram.

    Returns:
        spec_skewness: Returns the spectral skewness of a spectrogram or a signal.
    """

    spec_skewness = scipy.stats.skew(sig)

    return spec_skewness


def max(sig: np.ndarray) -> float:
    """
    Computes the maximum value of a signal.

    Args:
        sig: Input signal

    Returns:
        max_value: Returns the maximum value of a signal
    """

    return np.max(sig)


def min(sig: np.ndarray) -> float:
    """Computes the minimum value of a signal.

    Args:
        sig: Input signal

    Returns:
        min_value: Returns the minimum value of a signal
    """

    return np.min(sig)


def mean(sig: np.ndarray) -> float:
    """Computes the mean value of a signal.

    Args:
        sig: Input signal

    Returns:
        mean_value: Returns the mean value of a signal
    """

    return np.mean(sig)


def std(sig: np.ndarray) -> float:
    """Computes the standard deviation of a signal.

    Args:
        sig: Input signal

    Returns:
        std_value: Returns the standard deviation of a signal
    """

    return np.std(sig)


def kurtosis(sig: np.ndarray) -> float:
    """Computes the kurtosis of a signal.

    Args:
        sig: Input signal

    Returns:
        kurtosis Returns the kurtosis of a signal
    """

    return scipy.stats.kurtosis(sig)


def energy(sig: np.ndarray) -> float:
    """Computes the energy of a signal.

    Args:
        sig: Input signal

    Returns:
        energy: Returns the energy of a signal
    """

    return np.sum(sig**2) / len(sig)


def spectral_entropy(sig: np.ndarray) -> float:
    """Computes the spectral entropy of a signal.

    Args:
        sig (numpy array): Input signal

    Returns:
        spectral_entropy (float): Returns the spectral entropy of a signal
    """

    return scipy.stats.entropy(sig)


def flux(sig: np.ndarray) -> float:
    """Computes the flux of a signal.

    Args:
        sig: Input signal

    Returns:
        flux: Returns the flux of a signal
    """

    return librosa.onset.onset_strength(sig)


def rms_value(sig: np.ndarray) -> float:
    """Computes the Root Mean Square (RMS) of a signal.

    Args:
        sig: Input signal

    Returns:
        rms_value: Returns the root mean square of a signal
    """

    return float(np.sqrt(np.mean(np.square(sig))))


def spectral_centroid(sig: np.ndarray) -> float:
    """Computes the spectral centroid of a signal.

    Args:
        sig: Input signal

    Returns:
        spectral_centroid: Returns the spectral centroid of a signal
    """

    return float(np.mean(librosa.feature.spectral_centroid(sig)))


def envelope(sig: np.ndarray) -> np.ndarray:
    """Computes the envelope of a signal.

    Args:
        sig: Input signal

    Returns:
        envelope: Returns the envelope of a signal
    """

    return np.abs(signal.hilbert(sig))
