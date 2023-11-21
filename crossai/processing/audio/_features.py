import numpy as np
import librosa


def loudness(sig: np.ndarray) -> float:
    """Calculates the loudness of an audio signal.

    Args:
        sig: Input signal.

    Returns:
        float: Loudness
    """
    rms = np.sqrt(np.mean(sig**2))
    loudness = 20 * np.log10(rms)  # to db
    return loudness


def zero_crossing_rate(
    sig: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True
) -> np.ndarray:
    """Computes the zero-crossing rate of an audio time series.

    Args:
        sig: Input signal
        frame_length: Length of the frame over which to compute zero crossing rates.
        hop_length: Number of samples between successive frames.
        center: If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length].

    Returns:
        zero_crossing_rate: Returns the zero-crossing rate of an audio time series.
    """

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=sig, frame_length=frame_length, hop_length=hop_length, center=center)

    return zero_crossing_rate.flatten()
