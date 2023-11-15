import random
import numpy as np
import librosa
import nlpaug.augmenter.audio as naa


def loudness(sig, augment_times, factor_low, factor_high):
    """Changes the loudness of the signal.

    Args:
        sig (numpy array): Input signal.
        augment_times:
        factor_low (int): Factor to change the lower loudness of the signal.
        factor_high (int): Factor to change the upper loudness of the signal.

    Returns:
        loud_sig (numpy array): Returns the loudness changed signal.
    """
    loud_sig = []
    for i in range(augment_times):
        factor = tuple(random.uniform(factor_low, factor_high)
                       for i in range(2))
        aug = naa.LoudnessAug(factor=factor)
        loud_sig.append(np.array(aug.augment(sig)).flatten())
    return loud_sig


def time_stretch(signal, augment_times, factor):
    """Changes the speed of the signal.

    Args:
        signal (numpy array): Input signal.
        augment_times
        factor (float): Factor to change the speed of the signal.

    Returns:
        stretched_sig (numpy array): Returns the speed changed signal.
    """
    stretched_signal = []
    for i in range(augment_times):
        stretched_signal.append(librosa.effects.time_stretch(signal, factor))
        factor = np.random.uniform(0.1, 1)
    return stretched_signal


def pitch_shift(signal, augment_times, sr,  factor, zone=(0, 1), coverage=1):
    """
    Change the pitch of the signal

    Args:
        signal (numpy array): Input signal.
        augment_times:
        sr (int): Sampling rate of the input signal.
        factor (tuple): Factor to change the pitch of the signal.
        zone:
        coverage:

    Returns:
        pitched_sig (numpy array): Returns the pitch changed signal.
    """
    pitched_signal = []
    for i in range(augment_times):
        pitched_signal.append(naa.PitchAug(
            sampling_rate=sr, zone=zone, coverage=coverage,
            factor=factor).augment(signal)[0])
        factor = np.random.uniform(0, 50, size=2)
        factor = np.sort(factor)
    return pitched_signal
