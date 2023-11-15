import numpy as np
import warnings
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from crossai.processing import sliding_window_cpu
from crossai.pipelines.timeseries import TimeSeries


class Augmenter(BaseEstimator, TransformerMixin):
    """
    Class for custom augmenter creation.

    Args:
        func (function): Function to augment the data
        **kwargs: Keyword arguments for the augmenter function

    Returns:
        Augmenter object implementing the custom function on the data.

    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.augment_times = self.kwargs.get('augment_times', 0)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = []

        for i in range(len(X.data)):
            x = [self.func(X.data[i], **self.kwargs), X.labels[i]]
            if len(x[0]) > 0:
                for i in range(len(x[0])):
                    data.append([x[0][i], x[1]])

        return data
