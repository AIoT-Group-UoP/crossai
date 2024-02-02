import os
from functools import partial, wraps
from typing import Union
import pprint

import numpy as np

import pandas as pd

import scipy
from scipy import ndimage
from scipy import signal

import resampy
import librosa

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from . import _custom_processing, _custom_features
from ._utils import _transform_data, _custom_dimension_access, _generate_feature
from crossai import generate_transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


class Signal():
    """General time series analysis components and features.
    """

    def __init__(
        self,
        data: np.ndarray
    ):
        self.data = data
        self.X, self.Y = [], []
        self._keys = None

        if not hasattr(self, "features"):
            self.features = {}
        if not hasattr(self, "_len"):
            self._len = None
        self.function_transformer = {}

    def get_data(self, _=None) -> tuple:
        """Get new X and Y from the transformed data

        Returns:
            tuple: X and Y arrays
        """
        return self.X, self.Y

    def get_data_dict(self, _=None) -> tuple:
        """Get new X and Y from the transformed data

        Returns:
            tuple: X and Y arrays
        """
        return self.data

    def set_data(self, data: np.ndarray) -> None:
        """Set new X and Y from the transformed data

        """
        self.data = data
        self.__init__(data)

    def get_transformers(self, config) -> dict:
        return generate_transformers(self, config)

    def sliding_window_gpu(
        self,
        _=None,
        *,
        window_size: float,
        window_overlap: float,
        pad_end: bool,
        pad_value: float,
    ) -> None:
        """Apply a sliding window on the data. Overwrite the initial data with the windows.

        Args:
            window_size (float): Number of samples per window.
            window_overlap (float): Number of values to overlap with previous window.
            pad_end (bool): Pad at the end. If false last window is disgarded.
            pad_value (float): Value to pad the window.
        """
        #assert window_size < self.min_w, f"Window size upper limit must be min(n_samples)={self.min_w}"
        import tensorflow as tf
        _new_data = []
        for instance in self.data:
            if isinstance(instance['X'], dict):
                _tmp_data = []
                for _key, _value in instance['X'].items():
                    _tmp_data.append(tf.signal.frame(
                        _value, frame_length=window_size, frame_step=window_overlap, pad_end=pad_end, pad_value=0))
                _tmp_data = np.asarray(_tmp_data)

                for component in range(_tmp_data.shape[1]):
                    _tmp_dict = {'X': {}, 'Y': instance['Y']}
                    for _key, _value in zip(self._keys, _tmp_data[:, component, :]):
                        _tmp_dict['X'][_key] = _value
                    _new_data.append(_tmp_dict)

            elif isinstance(instance['X'], (list, np.ndarray)):
                _tmp_data = []
                for _component in instance['X']:
                    _tmp_data.append(tf.signal.frame(_component, frame_length=window_size, frame_step=window_overlap, pad_end=pad_end, pad_value=0))

                _tmp_data = np.asarray(_tmp_data[0])
                for window in range(_tmp_data.shape[1]):
                    _new_data.append({'X': _tmp_data[:, window], 'Y': instance['Y']})

        self.__init__(_new_data)


    def sliding_window_cpu(
        self,
        _=None,
        *,
        window_size: float,
        window_overlap: float,
    ) -> None:
        """Apply a sliding window on the data. Overwrite the initial data with the windows.

        Args:
            window_size (float): Number of samples per window.
            window_overlap (float): Number of values to overlap with previous window.
        """
        _new_data = []
        for instance in self.data:
            if isinstance(instance['X'], dict):
                _tmp_data = []
                for _key, _value in instance['X'].items():
                    _tmp_data.append(librosa.util.frame(
                        _value, frame_length=window_size, hop_length=window_overlap, axis=-1))
                _tmp_data = np.asarray(_tmp_data)
                _tmp_data = np.asarray(_tmp_data[0])
                _tmp_data = np.swapaxes(_tmp_data,1, -1)
                for component in range(_tmp_data.shape[1]):
                    _tmp_dict = {'X': {}, 'Y': instance['Y']}
                    for _key, _value in zip(self._keys, _tmp_data[:, component, :]):
                        _tmp_dict['X'][_key] = _value
                    _new_data.append(_tmp_dict)
            elif isinstance(instance['X'], (list, np.ndarray)):
                _tmp_data = []
                for _component in instance['X']:
                    _tmp_data.append(librosa.util.frame(
                        _component, frame_length=window_size, hop_length=window_overlap, axis=-1))
                _tmp_data = np.asarray(_tmp_data[0])
                _tmp_data = np.swapaxes(_tmp_data,1, -1)
                for window in range(_tmp_data.shape[1]):
                    _new_data.append({'X': _tmp_data[:, window], 'Y': instance['Y']})

        self.__init__(_new_data)


    def mean(self, _=None) -> None:
        """Create mean feature.
        """
        _generate_feature(self, 
            feature_naming='mean', feature_data=_custom_dimension_access(self.X, np.mean))

    def std(self, _=None) -> None:
        """Create standard deviation feature.
        """
        _generate_feature(self, 
            feature_naming='std', feature_data=_custom_dimension_access(self.X, np.std))

    def min(self, _=None) -> None:
        """Create min feature.
        """
        _generate_feature(self, 
            feature_naming='min', feature_data=_custom_dimension_access(self.X, np.min))

    def max(self, _=None) -> None:
        """Create max feature.
        """
        _generate_feature(self, 
            feature_naming='max', feature_data=_custom_dimension_access(self.X, np.max))

    def skewness(self, _=None) -> None:
        """Create skewness feature.
        """
        _generate_feature(self, feature_naming='skewness', feature_data=_custom_dimension_access(
            self.X, scipy.stats.skew))

    def kurtosis(self, _=None) -> None:
        """Create kurtosis feature.
        """
        _generate_feature(self, feature_naming='kurtosis', feature_data=_custom_dimension_access(
            self.X, scipy.stats.kurtosis))

    def flux(self, _=None) -> None:
        """Create fluctuation feature.
        """
        _generate_feature(self, feature_naming='flux', feature_data=_custom_dimension_access(
            self.X, scipy.stats.variation))

    def location_of_maximum(self, _=None) -> None:
        """Create location of maximum feature.
        """
        _generate_feature(self, feature_naming='location_of_maximum',
                               feature_data=_custom_dimension_access(self.X, np.argmax))

    def location_of_minimum(self, _=None) -> None:
        """Create location of minimum feature.
        """
        _generate_feature(self, feature_naming='location_of_minimum',
                               feature_data=_custom_dimension_access(self.X, np.argmin))

    def rms(self, _=None) -> None:
        """Create rms feature.
        """
        _generate_feature(self, 
            feature_naming='rms', feature_data=_custom_dimension_access(self.X, _custom_features.rms))

    def spectral_centroid(self, _=None, *, Fs: float) -> None:
        """Create spectral centroid feature.
        """
        _generate_feature(self, feature_naming='spectral_centroid', feature_data=_custom_dimension_access(
            self.X, partial(_custom_features.spectral_centroid, Fs=Fs)))

    def energy(self, _=None) -> None:
        """Create energy feature.
        """
        _generate_feature(self, 
            feature_naming='energy', feature_data=_custom_dimension_access(self.X, _custom_features.energy))

    def spectral_entropy(self, _=None) -> None:
        """Create spectral entropy feature.
        """
        _generate_feature(self, feature_naming='spectral_entropy', feature_data=_custom_dimension_access(
            self.X, partial(scipy.stats.entropy)))

    def fft(self, _=None) -> None:
        """Transform the data using fft.
        """
        _transform_data(self, partial(np.fft.fft))
        self.__init__(self.data)

    def ifft(self, _=None) -> None:
        """Get the inverse fft of the data.
        """
        _transform_data(self, partial(np.fft.ifft))
        self.__init__(self.data)

    def butterworth_filter(
        self,
        _=None,
        *,
        order: int,
        cutoff_freq: Union[int, list],
        type: str,
        output: str = 'sos',
        Fs: float
    ) -> None:
        """Apply the butterworth filter to all the data.

        Args:
            order (int): Order of the butterworth filter.
            cutoff_freq (Union[int, list]): Cutoff frequency or frequencies for band-pass.
            type (str): 'lowpass', 'bandpass', 'bandstop', 'highpass'.
            sampling_frequency (float): Sampling frequency.
            output (str, optional): Type of filters output, ['sos', 'ba']. Defaults to 'ba'.

        """
        _transform_data(self, partial(_custom_processing.butterworth_filtering,
                             order=order, cutoff_freq=cutoff_freq, type=type, Fs=Fs))
        self.__init__(self.data)

    def median_filter(self, _=None, *, kernel_size: int) -> None:
        """Apply median filter to the data.

        Args:
            kernel_size (int): Kernel size of the filter.
        """
        _transform_data(self, partial(signal.medfilt, kernel_size=kernel_size))
        self.__init__(self.data)

    def gaussian_filter(
        self,
        _=None,
        *,
        sigma: float,
        order: int,
        mode: str,
    ) -> None:
        """Apply gaussian filter on the data.

        Args:
            sigma (float): Standard deviation of the gaussian kernel.
            order (int): Gaussian's derivative oreder.
            mode (str): Mode to extend when overlap border.
        """
        _transform_data(self, 
            partial(_custom_processing.gaussian_filter, order=order, mode=mode, sigma=sigma))
        self.__init__(self.data)

    def savgol_filter(
        self,
        _=None,
        *,
        window_length: float,
        polyorder: int,
        deriv: int = 0,
        delta: float = 1.0,
        mode: str = 'interp',
        cval: float = 0.0
    ) -> None:
        """Apply Savitzky-Goal smoothing technique on the data.

        Args:
            window_length (float): Window to apply the filter.
            polyorder (int): Order of the polyonim
            deriv (int, optional): Polyonim derivative. Defaults to 0.
            delta (float, optional): The spacing of the samples to which the filter will be applied. This is only used if deriv > 0. Default is 1.0.
            mode (str, optional): Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This determines the type of extension to use for the padded signal to which the filter is applied. Defaults to 'interp'.
            cval (float, optional): Value to fill past the edges of the input if mode is 'constant'. Defaults to None.
        """
        assert mode in ['mirror', 'constant', 'nearest',
                        'wrap', 'interp'], "Mode is not correct"

        if mode == 'constant':
            assert cval, "Consant value cannot be None"

        _transform_data(self, partial(signal.savgol_filter, window_length=window_length,
                             polyorder=polyorder, deriv=deriv, delta=delta, mode=mode))
        self.__init__(self.data)

    def stft(
        self,
        _=None,
        *,
        Fs: float,
        window_length: int,
        overlap: int,
        scaling: str,
        return_onesided: bool = False,
        padded: bool = True,
    ) -> None:
        """Apply short-time fourier transform on the data.

        Args:
            Fs (float): Sampling frequency.
            window_length (int): Window size.
            overlap (int): Window's overlap.
            scaling (str): scaling method, 'psd' and 'spectrum'.
            return_onesided (bool, optional): Return only real part. Defaults to False.
            padded (bool, optional): Pad at the end. Defaults to True.
        """
        assert scaling in ['psd', 'spectrum'], "Scaling is not correct"

        _transform_data(self, partial(signal.stft, fs=Fs, nperseg=window_length, noverlap=overlap, padded=padded, return_onesided=return_onesided))
        self.__init__(self.data)

    def istft(
        self,
        _=None,
        *,
        Fs: float,
        window_length: int,
        overlap: int,
        scaling: str,
        input_onesided: bool = False,
    ) -> None:
        """Apply inverse short-time fourier transform on the data.

        Args:
            Fs (float): Sampling frequency.
            window_length (int): Window size.
            overlap (int): Window's overlap.
            scaling (str): scaling method, 'psd' and 'spectrum'.
            input_onesided (bool, optional): If input is only the real part. Defaults to False.
        """
        assert scaling in ['psd', 'spectrum'], "Scaling is not correct"

        _transform_data(self, partial(signal.istft, fs=Fs, nperseg=window_length, noverlap=overlap, input_onesided=input_onesided))
        self.__init__(self.data)

    def __stft_max_freq_ampl(
        self,
        x: np.ndarray,
        *,
        Fs: float,
        window_length: int,
        overlap: int,
        scaling: str,
        return_onesided: bool = False,
        padded: bool = True,
    ) -> Union[float, complex]:
        """Get the amplitude of the max frequence of a signal

        Args:
            x (np.ndarray): Input singal.
            Fs (float): Sampling frequency.
            window_length (int): Window size.
            overlap (int): Window overlap.
            scaling (str): Scaling mode, 'psd' or 'spectrum'.
            return_onesided (bool, optional): Return only real part of fft. Defaults to False.
            padded (bool, optional): Pad at the end. Defaults to True.

        Returns:
            (float, complex): Amplitude of max freq.
        """
        assert scaling in ['psd', 'spectrum'], "Scaling is not correct"

        _stft_gen = _transform_data(self, partial(signal.stft, fs=Fs, nperseg=window_length, noverlap=overlap, padded=padded, return_onesided=return_onesided), return_values=True)
        self.__init__(self.data)
        _feat = []
        for v in _stft_gen:
            _feat.append(v[-1][-1][-1])
        return np.asarray(_feat).flatten()

    def stft_max_freq_ampl(
        self,
        _=None,
        *,
        Fs: float,
        window_length: int,
        overlap: int,
        scaling: str,
        return_onesided: bool = False,
        padded: bool = True,
    ) -> None:
        """Get the amplitude of the max frequence of a signal

        Args:
            x (np.ndarray): Input singal.
            Fs (float): Sampling frequency.
            window_length (int): Window size.
            overlap (int): Window overlap.
            scaling (str): Scaling mode, 'psd' or 'spectrum'.
            return_onesided (bool, optional): Return only real part of fft. Defaults to False.
            padded (bool, optional): Pad at the end. Defaults to True.

        """
        _generate_feature(self, feature_naming='stft_max_freq_ampl', feature_data=_custom_dimension_access(self.X, partial(
            self.__stft_max_freq_ampl, Fs=Fs, window_length=window_length, overlap=overlap, scaling=scaling, return_onesided=return_onesided, padded=padded)))

    def resample(self, _=None, *, sr_origin: float, sr_target: float):
        """Upsample or downsample data.

        Args:
            sr_origin (float): Original sampling frequency of the signal.
            sr_target (float): Target sampling frequency of the singal
        """
        _transform_data(self, 
            partial(resampy.resample, sr_orig=sr_origin, sr_new=sr_target))
        self.__init__(self.data)



    def envelope(
        self,
        _=None,
        *,
        win_size: int,
        threshold: float,
    ) -> np.array:
        """ Envelope detection, cut signal based on a threshold of window's mean. 

        Args:
            time_series (Union[list, np.array]): Input signal.
            win_size (int): Window size.
            threshold (float): Cuttoff Threshold.

        Returns:
            np.array: Cut signal.
        """
        _transform_data(self, 
            partial(_custom_processing.envelope, win_size=win_size, threshold=threshold))
        self.__init__(self.data)
    
    def complex_to_real(self, _=None) -> None:
        """Convert complex to real.
        """
        _transform_data(self, partial(np.real))
        self.__init__(self.data)
    
    def pad(
        self, 
        _=None, 
        *, 
        direction='right', 
        mode='constant'
    ):
        """Pad data.

        Args:
            direction (str, optional): Padding direction. Defaults to 'right'.
            mode (str, optional): Padding mode. Defaults to 'constant'.
        """

        if self._len is None:
            self._len = self.len

        if self._len is not None:
            assert self.len <= self._len, "Input data is longer than provided"
            self.len = self._len
        _transform_data(self, partial(_custom_processing.pad, direction=direction, obj=self, mode=mode))
        self.__init__(self.data)

    def crop(
        self, 
        _=None
    ) -> None:
        """Crop signal to the number of samples of smallest instance.
        """

        _transform_data(self, partial(_custom_processing.crop,  obj=self))
        self.__init__(self.data)
    
    def amplify(self, _=None, *, change):
        _transform_data(self, partial(_custom_processing.amplify,  change=change))
        self.__init__(self.data)

    def abs(
        self, 
        _=None
    ) -> None:

        _transform_data(self, partial(abs))
        self.__init__(self.data)

    
    def get_features(self, _=None) -> np.ndarray:
        """Return features.

        Returns:
            dict: Instances features.
        """
        _features = []
        __len = 0
        for k, v in self.features.items():
            _tmp_feature_arr = []
            for feat_value in v.values():
                __len = len(feat_value)
                _tmp_feature_arr.append(feat_value)
            _features.append(_tmp_feature_arr)
        
        if __len > 1:
            return (np.swapaxes(np.asarray(_features), 0, 1), self.Y)
        else:
            return (np.asarray(_features).T[0], self.Y)
            

    @property
    def _pretty_print_features(self) -> None:
        pprint.pprint(self.features)

    @property
    def _pretty_print_data(self) -> None:
        pprint.pprint(self.data)
