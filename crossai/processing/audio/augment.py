import random
import numpy as np
import librosa
import nlpaug.augmenter.audio as naa


def loudness(sig, augment_times, factor_low, factor_high):
    """
    Change the loudness of the signal

    Args:
        sig (numpy array): Input signal
        sr (int): Sampling rate of the input signal
        factor (tuple): Factor to change the loudness of the signal

    Returns:
        loud_sig (numpy array): Returns the loudness changed signal
    """
    loud_sig = []
    for i in range(augment_times):
        factor = tuple(random.uniform(factor_low, factor_high)
                       for i in range(2))
        aug = naa.LoudnessAug(factor=factor)
        loud_sig.append(np.array(aug.augment(sig)).flatten())
    return loud_sig
