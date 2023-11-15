import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import librosa
import cv2


def resample_sig(sig, original_sr, target_sr):
    """ 
    Resample signal from its original sampling rate to a target sampling rate.

    Args: 
        sig: signal to resample
        original_sr: original sampling rate
        target_sr: target sampling rate

    Returns:
        rsmp_sig: resampled signal
    """

    secs = int(np.ceil(len(sig)/original_sr))  # seconds in signal
    samps = secs*target_sr     # num of samples to resample to
    rsmp_sig = scipy.signal.resample(sig, samps)
    return rsmp_sig


def amplify(sig, factor):
    """
    Amplify the signal by a factor.

    Args:
        sig (numpy array): Input signal
        factor (int): Factor to amplify the signal

    Returns:
        amp_sig (numpy array): Returns the amplified signal
    """
    amp_sig = sig * factor
    return amp_sig
