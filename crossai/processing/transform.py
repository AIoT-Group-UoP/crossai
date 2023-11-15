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


def complex_to_real(sig):
    """
    Make a signal from complex to real.

    Args:
        sig (numpy array): Input signal

    Returns:
        real_sig (numpy array): Returns the real signal
    """
    real_sig = np.real(sig)
    return real_sig


def fft(sig):
    """
    Compute the one-dimensional discrete Fourier Transform.

    Args:
        sig (numpy array): Input signal
        sr (int): Sampling rate of the input signal

    Returns:
        fft (numpy array): Returns the one-dimensional discrete
        Fourier Transform
    """
    fft = np.fft.fft(sig)
    return fft


def spec_to_rgb(spec, dsize=(256, 256), cmap='viridis'):
    """
    Convert a spectrogram to a 3 channel RGB image.

    Args:
        spec (numpy array): Input spectrogram   
        dsize (tuple): Size of the output image
        cmap (string): Colormap to use

    Returns:
        rgb_img (numpy array): Returns a 3 channel RGB image

    """

    norm = (spec-np.min(spec))/(np.max(spec)-np.min(spec))
    img = norm

    img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    rgba_img = plt.get_cmap(cmap)(img)
    rgb_img = np.delete(rgba_img, 3, 2)

    return rgb_img
