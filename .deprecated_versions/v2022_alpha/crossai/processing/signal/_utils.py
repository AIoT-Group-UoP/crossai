import numpy as np
from scipy import interpolate

def _transform_data(self, callback, *, return_values: bool = False) -> np.ndarray:
    """Apply transformation to the initial data and overwrite.

    Args:
        callback (function): Function to be applied.
        return_values (bool, optional): Yield reults. Defaults to False.

    Yields:
        np.ndarray: Transformation per instance.
    """
    _return_data = []
    for instance in self.data:
        if isinstance(instance['X'], dict):
            _new_value = []
            for _key, _value in instance['X'].items():
                _new_value.append(callback(_value))

            if callback.func.__name__ in ['stft', 'istft']:
                for idx, channel in enumerate(_new_value):
                    _new_value[idx] = _new_value[idx][-1]

            if not return_values:
                for idx, _key in enumerate(instance['X'].keys()):
                    instance['X'][_key] = np.real_if_close(
                        _new_value[idx], tol=1000)
            else:
                _return_data.append(_new_value)

        elif isinstance(instance['X'], (list,np.ndarray)):
            _new_value = []
            if len(np.asarray(instance['X']).shape) > 1:
                for channel in instance['X']:
                    _new_value.append(callback(channel))
            else:
                _new_value.append(callback(instance['X']))

            if callback.func.__name__ in ['stft', 'istft']:
                for idx, channel in enumerate(_new_value):
                    _new_value[idx] = _new_value[idx][-1]

            if not return_values:
                instance['X'] = np.real_if_close(_new_value, tol=1000)
            else:
                _return_data.append(_new_value[0])
    if return_values:
        return _return_data

def _custom_dimension_access(x: np.ndarray, callback) -> np.asarray:
    """Custom function to replicate numpy's axis=-1 argument. Used to access inner most dimension of dictionaries.

    Args:
        x (np.ndarray): Input data array.
        callback (function): Custom callback function to extract the feature.

    Returns:
        np.asarray: Feature's values.

    Example:
        >> signal = Signal(data)
        >> feature_data = signal._custom_dimension_access(data, np.argmin)

    """
    _feature = []
    for dim_1 in x:
        if len(np.asarray(dim_1).shape) >= 2:
            _tmp_feature = []
            for dim_2 in dim_1:
                _tmp_feature.append(callback(dim_2))
            _feature.append(_tmp_feature)
        else:
            try:
                _feature.append(callback(dim_1))
            except:
                _tmp_feature = []
                for dim_2 in dim_1:
                    _tmp_feature.append(callback(dim_2))
                _feature.append(_tmp_feature)

    return np.asarray(_feature, dtype=object)

def _generate_feature(
    self,
    *,
    feature_naming: str,
    feature_data: np.ndarray
) -> None:
    """Create feature dictionary.

    Args:
        feature_naming (str): Key that represents feature.
        feature_data (

    Example:
        >> signal = Signal(data)
        >> signal._generate_feature(feature_naming='min', feature_data = np.ndarray)
    """
    self.features[feature_naming] = {}
    _feature = feature_data
    if self._keys is None:
        for _instance_idx, _feature_value in enumerate(_feature):
            if isinstance(_feature_value, (np.ndarray, list)) and len(_feature_value) == 0:
                _feature_value = None
            self.features[feature_naming][_instance_idx] = _feature_value
    else:
        for _instance_idx in range(_feature.shape[0]):
            self.features[feature_naming][_instance_idx] = {}
            for _feature_value, _key in zip(_feature[_instance_idx], self._keys):
                if isinstance(_feature_value, (np.ndarray, list)) and len(_feature_value) == 0:
                    _feature_value = None
                self.features[feature_naming][_instance_idx][_key] = _feature_value


def interpolate_matrix(matrix, length, axis=0):
    """
    Produces a matrix of a desired size with interpolated values per column
    axis=0 or per row axis=1.
    Args:
        matrix:
        axis (int): The axis that the interpolation will be applied. A column
                    axis=0 / row axis=1.matrix (numpy.ndarray): An np.array
                    that the interpolation will be applied to.
        length (int):  The desired size of the matrix.

    Returns:
        matrix_interp (numpy.ndarray): The matrix with the interpolated values.
    """
    x = np.arange(0, matrix.shape[0])
    fit = interpolate.interp1d(x, matrix, axis=axis, kind="cubic")
    matrix_interp = fit(np.linspace(0, matrix.shape[0] - 1, length))

    return matrix_interp
