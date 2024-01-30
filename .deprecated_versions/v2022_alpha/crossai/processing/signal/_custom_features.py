import numpy as np
from scipy import signal

def spectral_centroid(x: np.ndarray, *, samplerate: float):
    magnitudes = np.abs(np.fft.rfft(x))
    print(samplerate)
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1])
    magnitudes = magnitudes[:length//2+1]
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)

def rms(x: np.ndarray) -> float:
    """Return rms of an array.

    Args:
        x (np.ndarray): Input array.

    Returns:
        float: RMS value.
    """
    return np.sqrt((np.linalg.norm(x, ord=2) ** 2)/len(x))
    
def energy(x: np.ndarray) -> float:
    """Energy of a signal.

    Args:
        x (np.ndarray): Input array.

    Returns:
        float: Energy value.
    """
    return np.sum(x ** 2)
