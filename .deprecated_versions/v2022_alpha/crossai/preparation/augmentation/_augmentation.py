import librosa
import numpy as np
import tsaug
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise, Dropout
import toml
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from . import _custom_augmentations
from ._custom_augmentations import Vtlp_Aug, Pitch_Aug, Roll_Aug


class Augmentation:

    def __init__(self) -> None:

        self.__part_assembler = []
    
    def __call__(self, config):
        for k, v in config.items():
            if k in dir(self):
                self.__getattribute__(k)(**v)
        self.__augmenter = np.sum(self.__part_assembler).augment
        return FunctionTransformer(self.augment_data)

    def time_wrap(self, n_times, prob, n_speed_change=3, max_speed_ratio=3.0):

        self.__part_assembler.append((TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio) * n_times) @ prob)

    def crop(self, n_times, size):
        self.__part_assembler.append((Crop(size) * n_times))

    def drift(self, n_times, prob, max_drift=0.5, n_drift_points=3, kind='additive', per_channel=True, normalize=True):

        self.__part_assembler.append((Drift(
            max_drift=max_drift, 
            n_drift_points=n_drift_points, 
            kind=kind, 
            per_channel=per_channel, 
            normalize=per_channel) * n_times) @ prob)
    
    def add_noise(self, n_times, prob, loc=0.0, scale=0.1, distr='gaussian', kind='additive', per_channel=True, normalize=True):
        self.__part_assembler.append((AddNoise(
            loc=loc,
            scale=scale,
            distr=distr,
            kind=kind, 
            per_channel=per_channel, 
            normalize=per_channel) * n_times) @ prob)

    def dropout(self, n_times, prob, p=0.05, size=1, fill='ffill', per_channel=False):
        self.__part_assembler.append((Dropout(
            p=p,
            size=size,
            fill=fill, 
            per_channel=per_channel) * n_times) @ prob)

    def quantize(self, n_times, prob, n_levels=10, how='uniform', per_channel=False):
        self.__part_assembler.append((Quantize(
            n_levels=n_levels,
            how=how,
            per_channel=per_channel) * n_times) @ prob)
        
    def vtlp(self, sr, prob):
        """
        Vtlp Augmentation function for audio data
        
        args: 
            sr : int sampling rate of the audio data
            prob : float probability of applying the augmentation
        """
        self.__part_assembler.append(Vtlp_Aug(sr=sr))
    
    def roll(self, prob ):
        """
        Roll Augmentation function for audio data

        args: 
            prob : float probability of applying the augmentation
        """
        self.__part_assembler.append(Roll_Aug())
        
    def pitch(self, sr, factor, prob):
        """
        Pitch Augmentation function for audio data

        args: 
            sr : int sampling rate of the audio data
            factor : float factor by which the pitch is to be augmented
            prob : float probability of applying the augmentation
        """
        self.__part_assembler.append(Pitch_Aug(sr=sr, factor=factor))

    def augment_data(self, data):
        data = list(data)
        x = data[0]
        if len(data) == 6:
            y = data[3]
        elif len(data) == 4:
            y = data[2]
        _new_data = []
        _y = []
        for instance in range(len(x)):
            _current = np.asarray(x[instance])
            _new_data.append(_current)
            n = self.__augmenter(x[instance])
            n = n.reshape(n.shape[0]//_current.shape[0], _current.shape[0], _current.shape[1])
            t_y = np.full(n.shape[0]+1, y[instance]).reshape(-1,1)
            [_new_data.append(i) for i in n]
            [_y.append(i) for i in t_y]

        data[0] = np.asarray(_new_data)
        if len(data) == 6:
            data[3] = _y
        elif len(data) == 4:
            data[2] = _y
        return tuple(data)