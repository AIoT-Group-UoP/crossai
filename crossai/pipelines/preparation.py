import numpy as np
import warnings
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


def augment_signal_data(crossai_object, augs):
    """
    Augment the data using the augmenters

    Args:
        crossai_object (CrossAI_Audio): CrossAI_Audio object
        augs (list): List of augmenters

    Returns:
        crossai_object (CrossAI_Audio): CrossAI_Audio object with augmented
                                            data
    """

    df_aug = pd.DataFrame(columns=['data', 'label'])
    df_aug['data'] = crossai_object.data
    df_aug['label'] = crossai_object.labels

    for i in range(len(augs)):
        x = augs[i].fit_transform(crossai_object)
        df_aug = pd.concat([df_aug, pd.DataFrame(
            x, columns=['data', 'label'])], ignore_index=True)

    crossai_object = TimeSeries(df_aug)

    return crossai_object


class Scaler(BaseEstimator, TransformerMixin):
    """
    Create a transformer class for the pipeline using a custom function

    Args:
        func (function): Function to be used in the pipeline
        **kwargs: Keyword arguments to be passed to the function
    """

    def __init__(self, scaler):
        """
        Initialize the Transformer class
        """
        self.scaler = scaler

    def fit(self, X, y=None):
        """
        Fit the transformer
        """
        return self

    def transform(self, X):
        """
        Transform the data using the custom function
        """

        X.data = self.scaler.fit_transform(X.data)

        return X
