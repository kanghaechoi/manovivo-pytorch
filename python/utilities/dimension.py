from typing import Union

import numpy as np


class Dimension:
    def __init__(self) -> None:
        pass

    def numpy_squeeze(
        self,
        array: np.ndarray,
        depth: int,
        width: int,
        height: int,
    ):
        array = array.reshape(((depth * width), height))

        return array

    def numpy_unsqueeze(
        self,
        array: np.ndarray,
        width: int,
        height: int,
        depth: int,
    ):
        array = array.reshape((depth, width, height))

        return array
