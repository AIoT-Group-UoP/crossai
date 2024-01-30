from typing import Union
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from crossai import generate_transformers
from ._utils import apply_to_splits, partial_fit, partial_transform

class Scaler:

    def __init__(
        self,
        *, 
        scale_axis: int = 1,
        partial: bool = False
    ) -> None:

        self.x_train, self.x_val, self.x_test = None, None, None
        self.scaler = None
        self._scale_axis = scale_axis
        self.partial = partial
        self.fit = True
    
    def transformers(self, config):
        return generate_transformers(self, config)
    
    def toggle_fit(self, data):
        self.fit = False
        return data

    def standard_scaler(
        self,
        data, 
        *, 
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:

        """Standardize features by removing the mean and scaling to unit variance.
           The standard score of a sample x is calculated as:
           z = (x - u) / s
           where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of 
           the training samples or one if with_std=False.
        Args:
            with_mean (bool, optional): If True, center the data before scaling. Defaults to True.
            with_std (bool, optional): If True, scale the data to unit variance (or equivalently, unit standard deviation). Defaults to True.
        """

        if not self.partial:
            return apply_to_splits(self, data, func=StandardScaler(with_mean=with_mean, with_std=with_std))
        else:
            if self.fit:
                return partial_fit(self, data, func=StandardScaler(with_mean=with_mean, with_std=with_std))
            else:
                return partial_transform(self, data, func=StandardScaler(with_mean=with_mean, with_std=with_std))

            
    def min_max_scaler(
        self, 
        data,
        *, 
        feature_range: tuple = (0, 1),
        clip: bool = False,
    ) -> None:

        """Transform features by scaling each feature to a given range.
           This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
           The transformation is given by:
           X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
           X_scaled = X_std * (max - min) + min
        Args:
            feature_range (tuple, optional): Desired range of transformed data. Defaults to (0, 1).
            clip (bool, optional): Set to True to clip transformed values of held-out data to provided feature range. Defaults to False.
        """
        if not self.partial:
            return apply_to_splits(self, data, func= MinMaxScaler(feature_range=feature_range, clip=clip))
        else:
            if self.fit:
                return partial_fit(self, data, func= MinMaxScaler(feature_range=feature_range, clip=clip))
            else:
                return partial_transform(self, data, func= MinMaxScaler(feature_range=feature_range, clip=clip))

    def max_abs_scaler(self, data) -> None:

        """Scale each feature by its maximum absolute value.
        """
        if not self.partial:
            return apply_to_splits(self, data, func= MaxAbsScaler())
        else:
            if self.fit:
                return partial_fit(self, data, func= MaxAbsScaler())
            else: 
                return partial_transform(self, data, func= MaxAbsScaler())

    @property
    def get_scaler(self): 
        return self.scaler

    @property
    def get_data(self): 
        return self.x_train, self.x_val, self.x_test