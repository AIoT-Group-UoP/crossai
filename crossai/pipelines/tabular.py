import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from crossai.processing.multiaxis._features import magnitude
from crossai.processing.signal._windowing import sliding_window_cpu
from crossai.processing.multiaxis._utils import axis_to_model_shape


class Tabular():
    """Tabular Class which contains the data, instances and labels of
            the multiaxial dataset.
    """

    def __init__(self, X):
        """Initialize the Tabular Class from data loaded using the
        multi_axis_data_loader_csv function.

        Args:
            X (pandas dataframe): Input data.

        Returns:
            self (CrossAI MultiAxisSignal Class): Returns an instance of the
            CrossAI MultiAxisSignal.
        """

        self.instance = X.instance
        self.labels = X.label
        self.feature = X.feature
        self.data = X.data


class MagnitudeExtractor(BaseEstimator, TransformerMixin):
    """Extract the magnitude of the data provided.

    Args:
        features (list): List with the list of features to extract the
            magnitude from and the name of the extracted feature. The list can
            contain multiple lists of features. For example:
            [[['acc_x', 'acc_y', 'acc_z'], 'acc_mag'],
            [['gyr_x', 'gyr_y', 'gyr_z'], 'gyr_mag']]
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        index_dict = {}
        for instance in X.instance.unique():
            indexes = X.instance[X.instance == instance].index
            for i in range(len(indexes)):
                index_dict[X.feature[indexes[i]]] = indexes[i]
            # compute the magnitude
            for feature in self.features:
                if feature[0][0] in index_dict.keys():
                    mag = magnitude(
                        *[X.data[index_dict[feat]] for feat in feature[0]])
                    X.instance = pd.concat([X.instance,
                                            pd.Series([instance])],
                                           ignore_index=True)
                    X.labels = pd.concat([X.labels,
                                          pd.Series([X.labels[indexes[0]]])],
                                         ignore_index=True)
                    X.feature = pd.concat([X.feature, pd.Series([feature[1]])],
                                          ignore_index=True)
                    X.data = pd.concat([X.data, pd.Series([mag])],
                                       ignore_index=True)
        return X


class MultiAxisSlidingWindow(BaseEstimator, TransformerMixin):
    """Create a sliding window of the motion data.

    Args:
        window_size (int): Size of the sliding window.
        step_size (int): Step size of the sliding window.
    """

    def __init__(self, window_size: int, step_size: int):
        self.window_size = window_size
        self.step_size = step_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(columns=['instance', 'feature', 'data', 'label'])
        Y = Tabular(df)
        Warning_shown = False

        for instance in X.instance.unique():
            indexes = X.instance[X.instance == instance].index
            # Check if the labels are array (pilot data)
            if not isinstance(X.labels[indexes[0]], str):
                windows = sliding_window_cpu(X.labels[indexes[0]],
                                             self.window_size,
                                             self.step_size,
                                             verbose=False)
                if windows is None:
                    continue
                # get the most frequent label in each window
                labels = np.array([])
                for i in range(len(windows)):
                    unique, counts = np.unique(windows[i],
                                               return_counts=True)
                    labels = np.append(labels, unique[np.argmax(counts)])
            # map each feature to its indexes
            for i in range(len(indexes)):
                data = sliding_window_cpu(X.data[indexes[i]],
                                          self.window_size,
                                          self.step_size,
                                          verbose=False)
                if data is None:
                    if not Warning_shown:
                        print("Error in sliding window instance. Probably "
                              "window size is bigger than the data or stride"
                              " is bigger than window size. Skipping instance."
                              " This warning will only be shown once.")
                        Warning_shown = True
                    continue
                Y_instance = []
                Y_labels = []
                Y_feature = []
                Y_data = []
                for j in range(len(data)):
                    # name instance "instance_{i}"
                    slided_instance = str(instance) + '_' + str(j)
                    Y_instance.append(slided_instance)
                    if not isinstance(X.labels[indexes[0]], str):
                        label = labels[j]
                    else:
                        label = X.labels[indexes[i]]
                    Y_labels.append(label)
                    Y_feature.append(X.feature[indexes[i]])
                    Y_data.append(data[j])
                Y.instance = pd.concat([Y.instance, pd.Series(Y_instance)],
                                       ignore_index=True)
                Y.labels = pd.concat([Y.labels, pd.Series(Y_labels)],
                                     ignore_index=True)
                Y.feature = pd.concat([Y.feature, pd.Series(Y_feature)],
                                      ignore_index=True)
                Y.data = pd.concat([Y.data, pd.Series(Y_data)],
                                   ignore_index=True)

        X.instance = Y.instance
        X.labels = Y.labels
        X.feature = Y.feature
        Y.data = np.array(Y.data.tolist())
        X.data = Y.data
        return X


class AxisToModelShape(BaseEstimator, TransformerMixin):
    """Function to convert multiple axes data to model shape
    (instance, window_size,features)

    Args:
        *kwargs: Each axis data/ feature.

    Returns:
        data (numpy array): Data in model shape.

    """
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Y_data = []
        Y_instance = []
        Y_labels = []
        Y_feature = []

        for instance in X.instance.unique():
            indexes = X.instance[X.instance == instance].index
            # we dont know how many features we have till we see the data
            data = axis_to_model_shape(*[X.data[indexes[i]]
                                         for i in range(len(indexes))])

            Y_data.append(data)
            Y_instance.append(instance)
            Y_labels.append(X.labels[indexes[0]])
            Y_feature.append(X.feature[indexes[0]])
        X.data = np.array(Y_data)
        X.instance = Y_instance
        X.labels = Y_labels
        X.feature = Y_feature
        return X