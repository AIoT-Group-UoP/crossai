import numpy as np
import scipy
import scipy.signal as signal


def butterworth_filter(
    sig: np.ndarray,
    f_type: str,
    sr: int,
    cutoff_low: int = 0,
    cutoff_high: int = 0,
    order: int = 4,
    output: str = 'ba'
) -> np.ndarray:
    """Applies a Butterworth filter.

    Given a signal, applies a Butterworth filter of type f_type and order order.

    Args:
        sig: Input signal
        f_type: Type of filter to apply, can be 'hp' (high pass), 'lp' (low pass) or 'bp' (band pass).
        sr: Sampling rate of the input signal.
        cutoff_low: Low cutoff frequency.
        cutoff_high: High cutoff frequency.
        order: Order of the filter.
        output: Type of output, can be 'sos' (second order sections) or 'ba' (numerator/denominator).

    Returns:
        filtered_sig: Returns the filtered signal.
    """
    if output == 'sos':
        if f_type == 'hp':
            sos = signal.butter(order, cutoff_low, 'hp', fs=sr,
                                output=output)
            filtered_sig = signal.sosfilt(sos, sig)
        elif f_type == 'lp':
            sos = signal.butter(order, cutoff_high, 'lp', fs=sr,
                                output=output)
            filtered_sig = signal.sosfilt(sos, sig)
        elif f_type == 'bp':
            sos = signal.butter(
                order, [cutoff_low, cutoff_high], 'bp', fs=sr,
                output=output)
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


def median_filter(
    sig: np.ndarray,
    kernel_size: int = None
) -> np.ndarray:
    """Applies a median filter to a nD signal.

    Args:
        sig: Input signal.
        kernel_size: Size of the kernel, must be an odd integer.

    Returns:
        filtered_sig: Returns the filtered signal.
    """

    filtered_sig = signal.medfilt(sig, kernel_size)
    return filtered_sig


def gaussian_filter(
    sig: np.ndarray,
    sigma: float = 1,
    order: int = 0,
    mode: str = 'wrap'
):
    """Applies gaussian filter on a signal.

    Args:
        sig: Input signal.
        sigma: Standard deviation of the gaussian kernel.
        order: Gaussian's derivative order.
        mode: Mode to extend when overlap border.

    Returns:
        Filtered signal.

    """

    return scipy.ndimage.gaussian_filter1d(sig, sigma=sigma,
                                           order=order, mode=mode)
