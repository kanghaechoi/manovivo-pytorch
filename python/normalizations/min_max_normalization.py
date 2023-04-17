import numpy as np

from normalizations.standard_normalization import StandardNormalization

from sklearn.preprocessing import MinMaxScaler


class MinMaxNormalization(StandardNormalization):
    def __init__(self, _features: np.ndarray) -> None:
        super().__init__(_features)

        scaler = MinMaxScaler()
        self.scaler = scaler.fit(_features)
