import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Audio():
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