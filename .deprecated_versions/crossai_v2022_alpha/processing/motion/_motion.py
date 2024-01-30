import sys
from functools import partial
import numpy as np
import scipy

from crossai.processing import Signal
from crossai.processing.signal._custom_processing import butterworth_filtering
from ..signal._utils import _transform_data


class Motion(Signal):
    
    def __init__(
        self, 
        data: dict
    ) -> None:
        super().__init__(data)

        self.X, self.Y, self._keys = [], [], []

        if self.data is not None:
            self.len = 0
            self.min_w = sys.maxsize
            for instance in self.data:
                self.Y.append(instance['Y'])
                instance_data = []
                for k,v in instance['X'].items():
                    if isinstance(instance['X'][k][0], (list, np.ndarray)) and len(instance['X'][k]) < 2:
                        instance['X'][k] = np.asarray(instance['X'][k][0])
                    else:
                        instance['X'][k] = np.asarray(instance['X'][k])
                    self.len = max(instance['X'][k].shape[-1], self.len)
                    self.min_w = min(instance['X'][k].shape[-1], self.min_w)
                    self._keys.append(k)
                    instance_data.append(instance['X'][k])
                self.X.append(instance_data)

            # Keep keys to map values to sensor readings
            self._keys = np.unique(self._keys)
            self.__acc_idx = 0
            self.X = np.asarray(self.X, dtype=object)
            self.Y = np.asarray(self.Y).reshape(-1, 1)

    def get_features(self, _=None) -> np.ndarray:
        """Get motion features.

        Returns:
            np.ndarray: Featurs array  and labels.
        """

        _n_features_per_component = len(self.features.keys())
        _n_instances = len(list(self.features.values())[0].keys())
        _n_components = len(list(list(self.features.values())[0].values())[0].keys())

        _features_per_component = list(self.features.keys())
        _components = list(list(list(self.features.values())[0].values())[0].keys())
        _features = []

        for i in range(_n_instances):
            __instance_arr = []
            for j in range(_n_components):
                __component_arr = []
                for k in range(_n_features_per_component):
                    __component_arr.append(self.features[_features_per_component[k]][i][_components[j]])
                __component_arr = np.hstack(np.asarray(__component_arr))
                __instance_arr.append(__component_arr)
            _features.append(np.asarray(__instance_arr))
        return np.asarray(_features), self.Y

    def __gravity_component_removal(self, x: np.ndarray, *, Fs: float) -> np.ndarray:
        """Remove gravity component from accelerometer readings callback function.

        Args:
            x (np.ndarray): Input signal.
            Fs (float): Sampling rate.

        Returns:
            np.ndarray: Acceleration siganl without gravity component.
        """
        if self.__acc_idx < 3:
            med_filt_values = scipy.signal.medfilt(x, kernel_size=3)
            _gravity_component = butterworth_filtering(med_filt_values, order=5, cutoff_freq=0.3, Fs=Fs, type='lowpass')
            self.__acc_idx += 1
            return np.asarray(x - _gravity_component, dtype=np.float32)
        elif self.__acc_idx == 5:
            self.__acc_idx = 0
            return x
        elif self.__acc_idx >= 3:
            return x

    def pure_acceleration(self, _=None, *, Fs: float, append: bool) -> None:
        """Remove gravity component from accelerometer readings.

        Args:
            Fs (float): Sampling rate.
            append (bool): If False replace the existent acc values.
        """
        _pure = _transform_data(self, partial(self.__gravity_component_removal, Fs=Fs), return_values=True)
        print(_pure[0])
        for instance in self.data:
            for i in range(3):
                if list(instance['X'].keys())[i].split('_')[0] == 'acc':
                    if append:
                       instance['X'][list(instance['X'].keys())[i] + '_pure'] = _pure[0][i]
                    else:
                        instance['X'][list(instance['X'].keys())[i]] = _pure[i]

        self.__init__(self.data)

    def cross_correlation(self, _=None) -> None:
        """Cross correlation feature between axes.
        """
        self.features['cross_correlation'] = {}
        for i in range(len(self.X)): self.features['cross_correlation'][i] = {}
        for instance_idx, instance in enumerate(self.X):
            for key, component in zip(self._keys, instance):
                __cross = []
                for cross_key, cross_component in zip(self._keys, instance):
                    if key != cross_key:
                        # self.features['cross_correlation'][instance_idx][f"{key}_{cross_key}"] = np.correlate(component, cross_component).item()
                        __cross.append(np.correlate(component, cross_component).item())
                    self.features['cross_correlation'][instance_idx][f"{key}"]= np.asarray(__cross)
                        
    def magnitude(self, _=None) -> None:
        """Magnitude of acceleration and gyroscope.
        """
        self.features['magnitude'] = {}
        for i in range(len(self.X)): self.features['magnitude'][i] = {}
        for instance_idx, instance in enumerate(self.X):
            for key, component in zip(self._keys, instance):
                try:
                    self.features['magnitude'][instance_idx][key.split('_')[0]] += np.sum(component ** 2)
                except KeyError:
                    self.features['magnitude'][instance_idx][key.split('_')[0]] = np.sum(component ** 2)
            for key, component in zip(self._keys, instance):
                self.features['magnitude'][instance_idx][key.split('_')[0]] = np.sqrt(self.features['magnitude'][instance_idx][key.split('_')[0]])
