import time
from sklearn.base import BaseEstimator, TransformerMixin


class DebugTransformer(TransformerMixin):
    """Transformer that prints the name of itself when called. Useful as an
        intermediate step in a pipeline to debug steps.
    """
    def __init__(self, name="DebugTransformer"):
        self.name = name
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("DebugTransformer: ", self.name)
        return X


class TimeMeasurementWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for sklearn transformers that prints the time a transformer
        takes to fit and transform.
    """
    def __init__(self, transformer: TransformerMixin):
        self.transformer = transformer

    def fit(self, X, y=None):
        start_time = time.time()
        self.transformer.fit(X, y)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Transformer {type(self.transformer).__name__} took "
              f"{execution_time:.2f} seconds to fit.")

        return self

    def transform(self, X):
        start_time = time.time()
        transformed_X = self.transformer.transform(X)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Transformer {type(self.transformer).__name__} took "
              f"{execution_time:.2f} seconds to transform.")

        return transformed_X
