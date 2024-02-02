import numpy as np
from crossai.processing import Signal


class TimeSeries(Signal):
    """General time series analysis components and features.
    """

    def __init__(
        self,
        data: dict
    ) -> None:

        super().__init__(data)

        if self.data is not None:
            self.X = []
            self.len = 0
            for i, instance in enumerate(self.data):

                instance['X'] = np.asarray(instance['X'])

                if len(instance['X'].shape) <= 2:
                    instance['X'] = np.asarray([instance['X']])

                elif len(np.asarray(instance['X']).shape) > 2:
                    instance['X'] = np.asarray(instance['X'])

                self.X.append(instance['X'][0])
                self.Y = np.vstack((self.Y, instance['Y'])) if i else instance['Y']
                self.len = max(self.len, np.asarray(instance['X']).shape[-1])
            try:
                self.X = np.asarray(self.X, dtype=object)
            except:
                pass
