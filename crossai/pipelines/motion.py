import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from crossai.processing.motion import pure_acceleration


class PureAccExtractor(BaseEstimator, TransformerMixin):
    """Extracts the pure acceleration from the data using a high-pass filter
        (removing very low frequency drifts or motion effects).

    Args:
        Fs (int): Sampling frequency of the data.
        acc_x (str): Name of the X dimension of the input signal.
        acc_y (str): Name of the Y dimension of the input signal.
        acc_z (str): Name of the Z dimension of the input signal.
        order (int): Order of the filter. Defaults to 2.
        fc (int): Cut-off frequency of the filter. Defaults to 1.
    """

    def __init__(self,
                 Fs: int,
                 acc_x: str = 'acc_x',
                 acc_y: str = 'acc_y',
                 acc_z: str = 'acc_z',
                 order: int = 2,
                 fc: int = 1):

        self.Fs = Fs
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.order = order
        self.fc = fc

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        index_dict = {}
        for instance in X.instance.unique():
            indexes = X.instance[X.instance == instance].index
            # map each feature to its indexes
            for i in range(len(indexes)):
                index_dict[X.feature[indexes[i]]] = indexes[i]
            pure_acc_x, pure_acc_y, \
                pure_acc_z = pure_acceleration(self.Fs,
                                               X.data[index_dict[self.acc_x]],
                                               X.data[index_dict[self.acc_y]],
                                               X.data[index_dict[self.acc_z]],
                                               self.order, self.fc)
            for axis in ['pure_acc_x', 'pure_acc_y', 'pure_acc_z']:
                X.instance = pd.concat([X.instance,
                                        pd.Series([instance])],
                                       ignore_index=True)
                X.labels = pd.concat([X.labels,
                                      pd.Series([X.labels[indexes[0]]])],
                                     ignore_index=True)
                X.feature = pd.concat([X.feature, pd.Series([axis])],
                                      ignore_index=True)
                X.data = pd.concat([X.data, pd.Series([eval(axis)])],
                                   ignore_index=True)

        return X
