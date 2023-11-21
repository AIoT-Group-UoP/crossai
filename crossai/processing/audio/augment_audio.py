import random
import numpy as np
import librosa
import nlpaug.augmenter.audio as naa


def loudness(
    sig: np.ndarray,
    augment_times: int,
    factor_low: int,
    factor_high: int
) -> list:
    """Changes the loudness of the signal.

    Args:
        sig: Input signal.
        augment_times: Number of times to augment original signal.
        factor_low: Factor to change the lower loudness of the signal.
        factor_high: Factor to change the upper loudness of the signal.

    Returns:
        loud_sig: Returns the loudness changed signal.
    """

    loud_sig = []
    for i in range(augment_times):
        factor = tuple(random.uniform(factor_low, factor_high)
                       for i in range(2))
        aug = naa.LoudnessAug(factor=factor)
        loud_sig.append(np.array(aug.augment(sig)).flatten())
    return loud_sig


def time_stretch(
    signal: np.ndarray,
    augment_times: int,
    factor: float
) -> list:
    """Changes the speed of the signal.

    Args:
        signal: Input signal.
        augment_times: Number of times to augment original signal.
        factor: Factor to change the speed of the signal.

    Returns:
        stretched_sig: Returns the speed changed signal.
    """

    stretched_signal = []
    for i in range(augment_times):
        stretched_signal.append(librosa.effects.time_stretch(signal, factor))
        factor = np.random.uniform(0.1, 1)
    return stretched_signal


def pitch_shift(
        signal: np.ndarray,
        augment_times: int,
        sr: int,
        factor: tuple,
        zone: tuple = (0, 1),
        coverage: int = 1):
    """
    Change the pitch of the signal

    Args:
        signal: Input signal.
        augment_times: Number of times to augment original signal.
        sr: Sampling rate of the input signal.
        factor: Factor to change the pitch of the signal.
        zone: Range of pitch shift. Default value is (0, 1).
        coverage: Percentage of audio to apply pitch shift. Default value is 1.

    Returns:
        pitched_sig: Returns the pitch changed signal.
    """

    pitched_signal = []
    for i in range(augment_times):
        pitched_signal.append(naa.PitchAug(
            sampling_rate=sr, zone=zone, coverage=coverage,
            factor=factor).augment(signal)[0])
        factor = np.random.uniform(0, 50, size=2)
        factor = np.sort(factor)
    return pitched_signal
