import numpy as np
import scipy
import scipy.signal as signal


def butterworth_filter(sig, f_type, sr, cutoff_low=0,
                       cutoff_high=0, order=4, output='ba'):
    """Applies a Butterworth filter to a 1D signal.

    Args:
        sig (numpy array): Input signal
        f_type (string): Type of filter to apply, can be 'hp' (high pass),
                            'lp' (low pass) or 'bp' (band pass).
        sr (int): Sampling rate of the input signal.
        cutoff_low (int): Low cutoff frequency.
        cutoff_high (int): High cutoff frequency.
        order (int): Order of the filter.
        output (string): Type of output, can be 'sos' (second order sections)
                                or 'ba' (numerator/denominator).

    Returns:
        filtered_sig (numpy array): Returns the filtered signal.
    """
    if output == 'sos':
        if f_type == 'hp':
            sos = signal.butter(order, cutoff_low, 'hp', fs=sr, output=output)
            filtered_sig = signal.sosfilt(sos, sig)
        elif f_type == 'lp':
            sos = signal.butter(order, cutoff_high, 'lp', fs=sr, output=output)
            filtered_sig = signal.sosfilt(sos, sig)
        elif f_type == 'bp':
            sos = signal.butter(
                order, [cutoff_low, cutoff_high], 'bp', fs=sr, output=output)
            filtered_sig = signal.sosfilt(sos, sig)
    elif output == 'ba':
        if f_type == 'hp':
            b, a = signal.butter(order, cutoff_low, 'hp', fs=sr)
            filtered_sig = signal.filtfilt(b, a, sig)
        elif f_type == 'lp':
            b, a = signal.butter(order, cutoff_high, 'lp', fs=sr)
            filtered_sig = signal.filtfilt(b, a, sig)
        elif f_type == 'bp':
            b, a = signal.butter(
                order, [cutoff_low, cutoff_high], 'bp', fs=sr)
            filtered_sig = signal.filtfilt(b, a, sig)
    else:
        raise ValueError("output must be 'sos' or 'ba'")

    return filtered_sig


def median_filter(sig, kernel_size=None):
    """Applies a median filter to a nD signal.

    Args:
        sig (numpy array): Input signal.
        kernel_size (int): Size of the kernel, must be an odd integer.

    Returns:
        filtered_sig (numpy array): Returns the filtered signal.
    """

    filtered_sig = signal.medfilt(sig, kernel_size)
    return filtered_sig


def gaussian_filter(sig, sigma=1, order=0, mode='wrap'):
    """Applies gaussian filter on a signal.

    Args:
        sig (numpy array): Input signal.
        sigma (float): Standard deviation of the gaussian kernel.
        order (int): Gaussian's derivative order.
        mode (str): Mode to extend when overlap border.

    Returns:
           np.ndarray: Filtered signal.

    """

    return scipy.ndimage.gaussian_filter1d(sig, sigma=sigma,
                                           order=order, mode=mode)
