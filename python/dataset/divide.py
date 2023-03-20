import math
import random

from typing import Optional, Tuple

import numpy as np


class Divide:
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = data
        self.labels = labels

        self.all_index = np.linspace(
            0,
            data.shape[0],
            data.shape[0],
            endpoint=False,
            dtype=int,
        )

        self.training_index: Optional[np.ndarray] = None
        self.test_index: Optional[np.ndarray] = None

    def fit(self, test_dataset_ratio: float = 0.2) -> None:
        number_of_data = self.data.shape[0]

        number_of_test_dataset = math.floor(number_of_data * test_dataset_ratio)

        np.random.seed(seed=192)
        self.test_index = np.random.choice(number_of_data, number_of_test_dataset)
        self.training_index = np.delete(self.all_index, self.test_index, axis=None)

    def training_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        training_data = self.data[self.training_index, :, :]
        training_labels = self.labels[self.training_index, :]

        return training_data, training_labels

    def test_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        test_data = self.data[self.test_index, :, :]
        test_labels = self.labels[self.test_index, :]

        return test_data, test_labels
