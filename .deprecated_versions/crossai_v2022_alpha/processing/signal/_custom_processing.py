from functools import wraps
import numpy as np
import librosa
from typing import Union

import pandas as pd

import scipy 

def butterworth_filtering(
    x: np.ndarray,
    *,
    order: int,
    cutoff_freq: Union[int, list],
    type: str,
    output: str = 'sos',
    Fs: float
) -> np.ndarray:
    """Signal Filtering using Butterworth Filter.

    Args:
        x (Union[list, np.ndarray]): Input singal.
        order (int): Order of the butterworth filter.
        cutoff_freq (Union[int, list]): Cutoff frequency or frequencies for band-pass.
        type (str): 'lowpass', 'bandpass', 'bandstop', 'highpass'.
        sampling_frequency (float): Sampling frequency.
        output (str, optional): Type of filters output, ['sos', 'ba']. Defaults to 'ba'.

    Returns:
        np.ndarray: Filtered signal.
    """
    init_shape = x.shape
    if isinstance(cutoff_freq, list) and len(cutoff_freq) > 2:
        return

    if isinstance(cutoff_freq, list) and type != 'bandpass' and len(cutoff_freq) == 2:
        output = 'bandpass'

    if len(x.shape) == 3 and x.shape[2] == 1:
        x = x.reshape(x.shape[:-1])
    if output == 'ba':
        numerator, denominator = scipy.signal.butter(
            order, cutoff_freq, type, output=output, fs=Fs)
        if len(x.shape) > 1 or not isinstance(x[0], np.ndarray):
            filtered_signal = scipy.signal.lfilt(numerator, denominator, x)
        elif len(x.shape) == 1:
            filtered_signal = []
            for signal in x:
                filtered_signal.append(scipy.signal.lfilt(
                    numerator, denominator, signal))
        filtered_signal = np.asarray(filtered_signal, dtype=np.float32)

        return filtered_signal.reshape(init_shape).astype(np.float32)

    elif output == 'sos':
        irr = scipy.signal.butter(
            order, cutoff_freq, type, output=output, fs=Fs)
        if len(x.shape) > 1 or not isinstance(x[0], np.ndarray):
            filtered_signal = scipy.signal.sosfiltfilt(irr, x, padlen=0)
        elif len(x.shape) == 1:
            filtered_signal = []
            for signal in x:
                filtered_signal.append(
                    scipy.signal.sosfiltfilt(irr, signal, padlen=0))
        filtered_signal = np.asarray(filtered_signal, dtype=np.float32)

        return filtered_signal.reshape(init_shape)

def gaussian_filter(
    x: np.ndarray,
    *,
    sigma: float,
    order: int,
    mode: str,
) -> np.ndarray:
    """Apply gaussian filter on a signal.

    Args:
        x (np.ndarray): Input signal.
        sigma (float): Standard deviation of the gaussian kernel.
        order (int): Gaussian's derivative oreder.
        mode (str): Mode to extend when overlap border.

    Returns:
        np.ndarray: Filtered signal.
    """
    return scipy.ndimage.gaussian_filter1d(x, sigma=sigma, order=order, mode=mode)

def envelope(
    x: np.ndarray,
    win_size: int,
    threshold: float,
) -> np.ndarray:
    """Envelope detection callback, cut signal based on a threshold of window's mean. 

    Args:
        time_series (np.ndarray): Input signal.
        win_size (int): Window size.
        threshold (float): Cuttoff Threshold.

    Returns:
        np.array: Cut signal.
    """
    mask = []
    x = np.asarray(x)
    y = pd.Series(x).apply(np.abs)
    y_mean = y.rolling(window=win_size, min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(1)
        else:
            mask.append(0)
    return np.multiply(x, mask)

def pad(
    x: np.ndarray,
    obj: object,
    *,
    direction: str = 'center',
    mode: str = 'constant'
) -> np.ndarray:
    """Pad sequencies to have equal shape.

    Args:
        x (np.ndarray): Input singal.
        obj (object): Module object.
        direction (str, optional): Padding direction. Defaults to 'center'.
        mode (str, optional): Type of padding. Defaults to 'constant'.

        'constant' (default)
        Pads with a constant value.

        'edge'
        Pads with the edge values of array.

        'linear_ramp'
        Pads with the linear ramp between end_value and the array edge value.

        'maximum'
        Pads with the maximum value of all or part of the vector along each axis.

        'mean'
        Pads with the mean value of all or part of the vector along each axis.

        'median'
        Pads with the median value of all or part of the vector along each axis.

        'minimum'
        Pads with the minimum value of all or part of the vector along each axis.

        'reflect'
        Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.

        'symmetric'
        Pads with the reflection of the vector mirrored along the edge of the array.

        'wrap'
        Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.

        'empty'
        Pads with undefined values.

    Returns:
        np.ndarray: _description_
    """

    assert mode in ['constant', 'median', 'reflect', 'maximum', 'mean', 'minimum', 'symmetric', 'wrap', 'edge', 'linear_ramp']

    
    new_x = []
    if len(np.asarray(x).shape) == 1:
        __x_len = np.asarray(x).shape[-1]
        if direction == 'center':
            __pad_size = (obj.len - __x_len)/2
            return np.pad(x, (int(np.floor(__pad_size)), int(np.ceil(__pad_size))), mode=mode)

        if direction == 'right':
            __pad_size = (obj.len - __x_len)
            return np.pad(x, (0, int(np.ceil(__pad_size))), mode=mode)

        if direction == 'left':
            __pad_size = (obj.len - __x_len)
            return np.pad(x, (int(np.floor(__pad_size)), 0), mode=mode)

    else:
        __axis =len(np.asarray(x).shape) - 2 if len(np.asarray(x).shape) > 2 else 0
        for i, _x in enumerate(np.rollaxis(np.asarray(x), __axis)):
            __x_len = np.asarray(x).shape[-1]
            if direction == 'center':
                __pad_size = (obj.len - __x_len)/2
                new_x.append(np.pad(_x, (int(np.floor(__pad_size)), int(np.ceil(__pad_size))), mode=mode))

            if direction == 'right':
                __pad_size = (obj.len - __x_len)
                new_x.append(np.pad(_x, (0, int(np.ceil(__pad_size))), mode=mode))

            if direction == 'left':
                __pad_size = (obj.len - __x_len)
                new_x.append(np.pad(_x, (int(np.floor(__pad_size)), 0), mode=mode))

        new_x = np.rollaxis(np.asarray(new_x), len(np.asarray(x).shape)-2)
    return new_x.astype(np.float32)

def crop(x:np.ndarray, obj: object) -> np.ndarray:
    if len(np.asarray(x).shape) == 2:
        return x[0][:obj.min_w]
    return x[:obj.min_w]


def amplify(x: np.ndarray, change: float) -> np.ndarray:

    x_db = librosa.amplitude_to_db(S=x)   
    x_db += change
    x_db = librosa.db_to_amplitude(x_db)

    return x_db

