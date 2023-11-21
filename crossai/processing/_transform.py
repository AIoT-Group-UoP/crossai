import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2


def resample_sig(
    sig: np.ndarray,
    original_sr: int,
    target_sr: int
) -> np.ndarray:
    """Resamples a signal.

    Given an input signal and it's sampling rate,
    resamples it to a target sampling rate.

    Args:
        sig: Signal to resample.
        original_sr: Original sampling rate.
        target_sr: Target sampling rate.

    Returns:
        rsmp_sig: Resampled signal.
    """

    secs = int(np.ceil(len(sig)/original_sr))
    samps = secs*target_sr
    rsmp_sig = scipy.signal.resample(sig, samps)

    return rsmp_sig


def amplify(
    sig: np.ndarray,
    factor: int
) -> np.ndarray:
    """Amplifies a signal.

    Given an input signal and a factor, amplifies the signal by the factor.

    Args:
        sig: Input signal.
        factor: Factor to amplify the signal.

    Returns:
        amp_sig: Returns the amplified signal.
    """

    amp_sig = sig * factor

    return amp_sig


def complex_to_real(sig: np.ndarray) -> np.ndarray:
    """Converts a complex signal to a real signal.

    Args:
        sig: Input signal.

    Returns:
        real_sig: Returns the real signal.
    """

    real_sig = np.real(sig)

    return real_sig


def fft(sig: np.ndarray) -> np.ndarray:
    """Compute the fft of a signal.

    Given an input signal,computes
    the one-dimensional discrete Fourier Transform.

    Args:
        sig: Input signal.

    Returns:
        fft: Returns the one-dimensional discreteFourier Transform.
    """

    fft = np.fft.fft(sig)

    return fft


def spec_to_rgb(
    spec: np.ndarray,
    dsize: tuple = (256, 256),
    cmap: str = 'viridis'
) -> np.ndarray:
    """Convert a Spectrogram to a RGB image.

    Given an input spectrogram, converts it to a 3 channel RGB image.

    Args:
        spec: Input spectrogram.
        dsize: Size of the output image.
        cmap: Colormap to use.

    Returns:
        rgb_img: Returns a 3 channel RGB image
    """

    norm = (spec-np.min(spec))/(np.max(spec)-np.min(spec))
    img = norm

    img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    rgba_img = plt.get_cmap(cmap)(img)
    rgb_img = np.delete(rgba_img, 3, 2)

    return rgb_img


def sliding_window_cpu(
    sig: np.ndarray,
    window_size: int,
    overlap: int,
    verbose: bool = True
) -> np.ndarray:
    """Applies a sliding window to a signal.

    Given an input signal,
    applies a sliding window to it to
    generate windows with overlap.

    Args:
        sig: Input signal.
        window_size: Window size in samples.
        overlap: Stride size in samples.
        verbose: Whether to print errors or not.

    Returns:
        sliding_window: Returns a sliding window of size window and
        overlap overlap.
    """

    overlap = window_size - overlap
    shape = sig.shape[:-1] + ((sig.shape[-1] - window_size + 1)//overlap,
                              window_size)
    strides = (sig.strides[0] * overlap,) + (sig.strides[-1],)

    try:
        return np.lib.stride_tricks.as_strided(sig, shape=shape,
                                               strides=strides)
    except ValueError:
        if verbose:
            print("Error in sliding window instance."
                  " Probably window size is bigger than the data or stride is"
                  " bigger than window size. Returning None.")

        return None
