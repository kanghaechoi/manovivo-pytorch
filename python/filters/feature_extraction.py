import numpy as np


class FeatureExtraction:
    def __init__(self, _input: np.ndarray) -> None:
        self.input = _input

    def angle(self, column_index) -> np.ndarray:
        angle_array = np.zeros((1, 2))

        input_dataframe = pd.DataFrame(self.input)[col]

        angle_temp = np.array(pd.to_numeric(input_dataframe))
        angle = angle_temp.reshape((-1, 1))
        # x_angle_transformed = zero_to_one(x_angle)
        # x_angle_mean, x_angle_std = mean_std(x_angle_transformed)
        angle_mean, angle_std = mean_std(angle)
        angle_array[0, 0] = angle_mean
        angle_array[0, 1] = angle_std

        return angle_array
