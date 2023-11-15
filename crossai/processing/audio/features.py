import numpy as np
import librosa


def loudness(sig):
    """Calculates the loudness of an audio signal.

    Args:
        sig (np.ndarray): Input signal.

    Returns:
        float: Loudness
    """
    rms = np.sqrt(np.mean(sig**2))
    loudness = 20 * np.log10(rms)  # to db
    return loudness


def zero_crossing_rate(sig, frame_length=2048, hop_length=512, center=True):
    """Computes the zero-crossing rate of an audio time series.

    Args:
        sig (numpy array): Input signal
        frame_length (int): Length of the frame over which to compute zero 
            crossing rates.
        hop_length (int): Number of samples between successive frames.
        center (bool): If True, the signal y is padded so that frame D[:, t]
            is centered at y[t * hop_length].

    Returns:
        zero_crossing_rate (numpy array): Returns the zero-crossing rate 
            of an audio time series.
    """

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=sig, frame_length=frame_length, hop_length=hop_length, center=center)

    return zero_crossing_rate.flatten()
