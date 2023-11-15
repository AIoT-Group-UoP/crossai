import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from crossai.processing._utils import pad_or_trim
from crossai.processing import sliding_window_cpu


class TimeSeries():
    """CrossAI TimeSeries object that contains the data and labels of a time
            series dataset.
    """

    def __init__(self, X):
        """Initializes the CrossAI TimeSeries object.

        Args:
            X (pandas dataframe): Input data

        Returns:
            self (CrossAI Signal Class): Returns an instance of the
            CrossAI Signal Class
        """

        self.data = X.data
        self.labels = X.label


class ToPandas(BaseEstimator, TransformerMixin):
    """CrossAI Pipeline object that returns pipeline data as pandas Dataframe.

    Args:
        func (function): Function to be used in the pipeline
        **kwargs: Keyword arguments to be passed to the function
    """

    def __init__(self):
        """Initializes the Transformer class.
        """
        return None

    def fit(self, X, y=None):
        """Fits the transformer.
        """
        return self

    def transform(self, X):
        """Transforms the data using the custom function.
        """

        if type(X.data) is np.ndarray:
            Y = []

            for i in range(len(X.data)):
                Y.append((X.data[i]))
            X.data = Y

        X.data = pd.Series(X.data)
        X.labels = pd.Series(X.labels)

        return X


class Transformer(BaseEstimator, TransformerMixin):
    """Create a transformer object for the pipeline using a custom function.

    Args:
        func (function): Function to be used in the pipeline
        **kwargs: Keyword arguments to be passed to the function
    """

    def __init__(self, func, **kwargs):
        """Initializes the Transformer class.
        """
        self.func = func
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Fits the transformer.
        """
        return self

    def transform(self, X):
        """Transforms the data using the custom function.
        """

        Y = []

        for i in range(len(X.data)):
            Y.append(self.func(X.data[i], **self.kwargs))

        X.data = Y

        return X


class PadOrTrim(BaseEstimator, TransformerMixin):
    """Creates a transformer object for the pipeline using a custom function.

    Args:
        func (function): Function to be used in the pipeline
        **kwargs: Keyword arguments to be passed to the function
    """

    def __init__(self, **kwargs):
        """
        Initialize the Transformer class
        """

        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Fit the transformer
        """
        return self

    def transform(self, X):
        """
        Transform the data using the custom function
        """

        X.data = pad_or_trim(X.data, **self.kwargs)

        return X


class SlidingWindow(BaseEstimator, TransformerMixin):
    """Performs sliding window procedure onto the data.

    Useful in time series analysis to convert a sequence of objects (scalar or
    array-like) into a sequence of windows on the original sequence. Each
    window stacks together consecutive objects, and consecutive windows are
    separated by a constant stride.

    Args:
        window_size: int, optional, default: ``10``
            Size of each sliding window.
        overlap: int, optional, default: ``1``

    """

    def __init__(self, window_size=10, overlap=1):
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Slide windows over X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        X : ndarray of shape (n_windows, size, ...)
            Windows of consecutive entries of the original time series.
        """

        Y = []
        Z = []

        for i in range(len(X.data)):
            Y.append(sliding_window_cpu(X.data[i], self.window_size,
                                        self.overlap))
            Z.append(np.repeat(X.labels[i], len(Y[i])))

        X.data = list(np.concatenate(Y, axis=0))
        X.labels = list(np.concatenate(Z, axis=0))

        return X
