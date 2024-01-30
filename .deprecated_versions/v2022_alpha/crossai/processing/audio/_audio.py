import os
import sys
from functools import partial
from typing import Union

import numpy as np

import librosa


from crossai.processing import Signal
from . import _custom_features
from ..signal._utils import _transform_data, _generate_feature, _custom_dimension_access

sys.path.append(os.getcwd())


class Audio(Signal):

    def __init__(
        self,
        data: Union[list, np.ndarray]
    ) -> None:
        """Audio module constructor. Corrects data shape.

        Args:
            data (list, np.ndarray): Input data.
        """
        super().__init__(data)

        if self.data is not None:
            self.X = []
            self.len = 0
            self.min_w = 10e+1000000000000000
            for i, instance in enumerate(self.data):

                instance['X'] = np.asarray(instance['X'])

                if len(instance['X'].shape) <= 2:
                    instance['X'] = np.asarray([instance['X']])

                elif len(np.asarray(instance['X']).shape) > 2:
                    instance['X'] = np.asarray(instance['X'])

                
                self.X.append(instance['X'][0])
                self.Y = np.vstack((self.Y, instance['Y'])) if i else [instance['Y']]
                self.len = max(self.len, np.asarray(instance['X']).shape[-1])
                self.min_w = min(self.min_w, np.asarray(instance['X']).shape[-1])
            try:
                self.X = np.asarray(self.X, dtype=object)
            except:
                pass

    def convert_to_mono(self, _=None) -> None:
        """Convert multi-channel audio to mono.
        """
        for instance in self.data:
            instance['X'] = librosa.to_mono(instance['X'].astype(np.float32))
        self.__init__(self.data)

    def mfccs(
        self,
        _=None,
        *,
        Fs: float,
        n_mfcc: int,
        dct_type: int,
        norm: str,
        lifter: int
    ) -> None:
        """Get Mel frequencies cepstral coefficients.

        Args:
            Fs (float): Sampling rate.
            n_mfcc (int): Number of mels.
            dct_type (int): Discrete cosine transform (DCT) type {1,2,3}.
            norm (str): If dct_type is 2 or 3, setting norm='ortho' uses an ortho-normal DCT basis. Normalization is not supported for dct_type=1.
            lifter (int): Setting lifter >= 2 * n_mfcc emphasizes the higher-order coefficients. As lifter increases, the coefficient weighting becomes approximately linear.
        """

        _transform_data(self, partial(librosa.feature.mfcc,
                             sr=Fs, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter))
        self.__init__(self.data)

    def melspectrogram(
        self,
        _=None,
        *,
        Fs: float,
        n_fft: int,
        hop_length: int,
        win_length: int = None,
        window: str,
        center: bool,
        pad_mode: str,
        power: float,
        fmax : float
    ) -> None:
        """Get log-mel spectrograms of the audio.

        Args:
            Fs (float): Sampling rate.
            n_fft (int): Number of FFTs.
            hop_length (int): Window hop size.
            win_length (int): Window's size.
            window (str): Window type.
            center (bool): Centering the frame.
            pad_mode (str): Pad at the end.
            power (float): Exponent for the magnitude melspectrogram.
            fmax (float): Maximum frequency.
        """

        
        
        _transform_data(self, partial(librosa.feature.melspectrogram,
                             sr=Fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, power=power, fmax=fmax))
        self.__init__(self.data)
        
    
    def log_melspectrogram(
        self,
        _=None,
        *,
        Fs: float,
        n_fft: int,
        hop_length: int,
        win_length: int = None,
        window: str,
        center: bool,
        pad_mode: str,
        power: float,
        fmax : float
    ) -> None:
        """Get log-mel spectrograms of the audio.

        Args:
            Fs (float): Sampling rate.
            n_fft (int): Number of FFTs.
            hop_length (int): Window hop size.
            win_length (int): Window's size.
            window (str): Window type.
            center (bool): Centering the frame.
            pad_mode (str): Pad at the end.
            power (float): Exponent for the magnitude melspectrogram.
            fmax (float): Maximum frequency.
        """

        
        
        _transform_data(self, partial(_custom_features.logmelspectogram, Fs=Fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, power=power, fmax=fmax))
        self.__init__(self.data)
        
    def melspectogram_denoised(
        self,
        _=None,
        *,
        Fs: float,
        n_fft: int,
        hop_length: int,
        win_length: int = None,
        window: str,
        center: bool,
        pad_mode: str,
        fmax : float,
        power_noise : float,
        power_signal : float
    ) -> None:
        """Get log-mel spectrograms of the audio.

        Args:
            Fs (float): Sampling rate.
            n_fft (int): Number of FFTs.
            hop_length (int): Window hop size.
            win_length (int): Window's size.
            window (str): Window type.
            center (bool): Centering the frame.
            pad_mode (str): Pad at the end.
            power (float): Exponent for the magnitude melspectrogram.
            fmax (float): Maximum frequency.
        """
        
        
        _transform_data(self, partial(_custom_features.melspectogram_power_filtering, Fs=Fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode,fmax=fmax, power_noise=power_noise, power_signal=power_signal))
        self.__init__(self.data)

    def inverse_melspectrogram(
        self,
        _=None,
        *,
        Fs: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        center: bool,
        pad_mode: str,
        power: float
    ) -> None:
        """Get audio time series from mel-spectrograms.

        Args:
            Fs (float): Sampling rate.
            n_fft (int): Number of FFTs.
            hop_length (int): Window hop size.
            win_length (int): Window's size.
            window (str): Window type.
            center (bool): Centering the frame.
            pad_mode (str): Pad at the end.
            power (float): Exponent for the magnitude melspectrogram.
        """
        _transform_data(self, partial(librosa.feature.inverse.mel_to_audio,
                             sr=Fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, power=power))
        self.__init__(self.data)

    def q_transform(
        self,
        _=None,
        *,
        Fs: float,
        n_bins: int,
        bins_per_octave: int,
        fmin: float,
        hop_length: int,
        norm: int,
    ) -> None:
        """Compute the constant-Q transform of an audio signal.

        Args:
            Fs (float): Sampling rate.  
            n_bins (int): Number of frequency bins, starting at fmin.
            bins_per_octave (int): Number of bins per octave
            fmin (float): Minimum frequency.
            hop_length (int): number of samples between successive CQT columns.
            norm (int): Type of norm to use for basis function normalization.
        """
        _transform_data(self, partial(librosa.core.cqt,
                                     sr=Fs, n_bins=n_bins, hop_length=hop_length, fmin=fmin, norm=norm, bins_per_octave=bins_per_octave))
        self.__init__(self.data)

    def zero_crossing_rate(
        self,
        _=None,
        *,
        frame_length: int,
        hop_length: int,
        center: bool = False,
    ) -> None:
        """Compute the zero-crossing rate of an audio time series.

        Args:
            frame_length (int): Length of the frame over which to compute zero crossing rates
            hop_length (int): Number of samples to advance for each frame
            _ (_type_, optional): _description_. Defaults to None.
            center (bool, optional): If True, frames are centered by padding the edges of y. Defaults to False.
        """
        _generate_feature(self, feature_naming='zero_crossing_rate', feature_data=_custom_dimension_access(self.X, partial(librosa.feature.zero_crossing_rate, frame_length=frame_length, hop_length=hop_length, center=center)))

    def spectral_kurtosis(self, _=None) -> None:
        """Get spectral kurtosis for all instances.
        """
        _generate_feature(self, feature_naming='spectral_kurtosis', feature_data=_custom_dimension_access(
            self.X, partial(_custom_features.spectral_kurtosis)))

    def spectral_skewness(self, _=None) -> None:
        """Get spectral skewness for all instances.
        """
        _generate_feature(self, feature_naming='spectral_skewness', feature_data=_custom_dimension_access(
            self.X, partial(_custom_features.spectral_skewness)))

    def loudness(
        self,
        _=None
    ) -> None:
        """Get loudness for all instances.
        """
        _generate_feature(self, feature_naming='loudness', feature_data=_custom_dimension_access(
            self.X, partial(_custom_features.loudness)))
    
    def chroma(
        self,
        _=None,
        *,
        n_chroma: int,
        hop_length: int,
        fmin: float,
        Fs: float,
        norm: int,
        threshold: float
    ) -> None:
        """Constant-Q chromagram.

        Args:
            n_chroma (int): Number of chroma bins to produce.
            hop_length (int): Number of samples between successive chroma frames.
            fmin (float): Minimum frequency to analyze in the CQT.
            Fs (float): Sampling rate.
            norm (int): Column-wise normalization of the chromagram.
            threshold (float): Pre-normalization energy threshold. Values below the threshold are discarded, resulting in a sparse chromagram.
        """

        _transform_data(self, partial(librosa.feature.chroma_cqt,
                                     sr=Fs, n_chroma=n_chroma, hop_length=hop_length, fmin=fmin, norm=norm, threshold=threshold, bins_per_octave=None))
        self.__init__(self.data)

    def chroma_cens(
        self,
        _=None,
        *,
        n_chroma: int,
        hop_length: int,
        fmin: float,
        Fs: float,
        norm: int,
    ) -> None:
        """Computes the chroma variant “Chroma Energy Normalized” (CENS).

        Args:
            n_chroma (int): Number of chroma bins to produce.
            hop_length (int): Number of samples between successive chroma frames.
            fmin (float): Minimum frequency to analyze in the CQT.
            Fs (float): Sampling rate.
            norm (int): Column-wise normalization of the chromagram.
            _ (_type_, optional): _description_. Defaults to None.
        """

        _transform_data(self, partial(librosa.feature.chroma_cens,
                                     sr=Fs, n_chroma=n_chroma, hop_length=hop_length, fmin=fmin, norm=norm, bins_per_octave=None))
        self.__init__(self.data)

    def chroma_stft(
        self,
        _=None,
        *,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_chroma: int,
        center: bool,
        pad_mode: str = 'constant',
        Fs: float,
        norm: int,
    ) -> None:
        """Compute a chromagram from a waveform or power spectrogram.

        Args:
            n_fft (int): FFT window size.
            win_length (int): Number of samples per window.
            hop_length (int): Hop length.
            n_chroma (int): Number of chroma bins to produce.
            center (bool): Frame is centered.
            Fs (float): Sampling rate.
            norm (int): Normalization factor for each filter.
            pad_mode (str, optional): If center=True, the padding mode to use at the edges of the signal. Defaults to 'constant'.
        """

        _transform_data(self, partial(librosa.feature.chroma_stft,
                                     sr=Fs, n_fft=n_fft, win_length=win_length, n_chroma=n_chroma, hop_length=hop_length, center=center, norm=norm, pad_mode=pad_mode))
        self.__init__(self.data)
        
