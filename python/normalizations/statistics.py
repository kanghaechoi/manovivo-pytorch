import numpy as np

from sklearn.preprocessing import StandardScaler


class Statistics:
    def __init__(self, _features: np.ndarray) -> None:
        scaler = StandardScaler()
        self.scaler = scaler.fit(_features)

    def transform(self, _features: np.ndarray) -> np.ndarray:
        normalized_array = self.scaler.transform(_features)

        return normalized_array

    def mean(self) -> np.ndarray:
        _mean = self.scaler.mean_

        return _mean

    def standard_deviation(self) -> np.ndarray:
        _variance = self.scaler.var_
        _standard_deviation = np.sqrt(_variance)

        return _standard_deviation
