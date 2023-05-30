from typing import Optional, Tuple

import numpy as np

from scipy import integrate


class Extraction:
    def __init__(
        self,
        _research_question: int,
        _ages: list,
        _data_chunks: zip,
        _sample_length: int,
        _authentication_classes: Optional[list] = None,
    ) -> None:
        self.research_question = _research_question
        self.ages = _ages
        self.data_chunks = _data_chunks
        self.sample_length = _sample_length
        self.authentication_classes = _authentication_classes

    def _create_array_default(self, data: np.ndarray) -> np.ndarray:
        array = data.reshape(1, -1)

        return array

    def _create_array_for_integral(self, data: np.ndarray) -> np.ndarray:
        data = data.reshape(1, -1)

        integral_data = integrate.cumtrapz(data)

        last_integral_data = np.expand_dims(integral_data[:, -1], axis=0)

        array = np.hstack((integral_data, last_integral_data))

        return array

    def extract_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        is_first_labelset: bool = True
        is_first_dataset: bool = True

        for chunk_index, (hand_data, wrist_data, helical_data, labels) in enumerate(
            self.data_chunks
        ):
            number_of_samples = min(
                len(hand_data),
                len(wrist_data),
                len(helical_data),
            )

            if is_first_labelset is True:
                labelset = np.ones((number_of_samples, 1), dtype=int) * labels
                is_first_labelset = False
            else:
                _labels = np.ones((number_of_samples, 1), dtype=int) * labels
                labelset = np.concatenate((labelset, _labels), axis=0)

            is_first_feature_stack: bool = True

            for sample_index in range(number_of_samples):
                hand_array = np.array(hand_data[sample_index]).astype(float)
                hand_array = hand_array.transpose()

                wrist_array = np.array(wrist_data[sample_index]).astype(float)
                wrist_array = wrist_array.transpose()

                helical_array = np.array(helical_data[sample_index]).astype(float)
                helical_array = helical_array.transpose()

                hand_x_angle = self._create_array_default(hand_array[0, :])
                hand_y_angle = self._create_array_default(hand_array[1, :])
                hand_z_angle = self._create_array_default(hand_array[2, :])

                thumb_x_angle = self._create_array_default(hand_array[3, :])
                index_x_angle = self._create_array_default(hand_array[4, :])

                thumb_x_acceleration = self._create_array_default(hand_array[6, :])
                thumb_x_velocity = self._create_array_for_integral(hand_array[6, :])
                thumb_y_acceleration = self._create_array_default(hand_array[7, :])
                thumb_y_velocity = self._create_array_for_integral(hand_array[7, :])
                thumb_z_acceleration = self._create_array_default(hand_array[8, :])
                thumb_z_velocity = self._create_array_for_integral(hand_array[8, :])

                index_x_acceleration = self._create_array_default(hand_array[9, :])
                index_x_velocity = self._create_array_for_integral(hand_array[9, :])
                index_y_acceleration = self._create_array_default(hand_array[10, :])
                index_y_velocity = self._create_array_for_integral(hand_array[10, :])
                index_z_acceleration = self._create_array_default(hand_array[11, :])
                index_z_velocity = self._create_array_for_integral(hand_array[11, :])

                wrist_x_angle = self._create_array_default(wrist_array[0, :])
                wrist_y_angle = self._create_array_default(wrist_array[1, :])
                wrist_z_angle = self._create_array_default(wrist_array[2, :])

                wrist_x_acceleration = self._create_array_default(wrist_array[3, :])
                wrist_x_velocity = self._create_array_for_integral(wrist_array[3, :])
                wrist_y_acceleration = self._create_array_default(wrist_array[4, :])
                wrist_y_velocity = self._create_array_for_integral(wrist_array[4, :])
                wrist_z_acceleration = self._create_array_default(wrist_array[5, :])
                wrist_z_velocity = self._create_array_for_integral(wrist_array[5, :])

                helical_x_angle = self._create_array_default(helical_array[0, :])
                helical_y_angle = self._create_array_default(helical_array[1, :])
                helical_z_angle = self._create_array_default(helical_array[2, :])

                feature = np.concatenate(
                    (
                        hand_x_angle,
                        hand_y_angle,
                        hand_z_angle,
                        thumb_x_angle,
                        index_x_angle,
                        thumb_x_acceleration,
                        thumb_y_acceleration,
                        thumb_z_acceleration,
                        thumb_x_velocity,
                        thumb_y_velocity,
                        thumb_z_velocity,
                        index_x_acceleration,
                        index_y_acceleration,
                        index_z_acceleration,
                        index_x_velocity,
                        index_y_velocity,
                        index_z_velocity,
                        wrist_x_angle,
                        wrist_y_angle,
                        wrist_z_angle,
                        wrist_x_acceleration,
                        wrist_y_acceleration,
                        wrist_z_acceleration,
                        wrist_x_velocity,
                        wrist_y_velocity,
                        wrist_z_velocity,
                        helical_x_angle,
                        helical_y_angle,
                        helical_z_angle,
                    )
                )

                if is_first_feature_stack is True:
                    data = feature
                    is_first_feature_stack = False
                else:
                    data = np.dstack((data, feature))

                if sample_index == number_of_samples - 1:
                    is_first_feature_stack = True

                print(
                    "Subject {:d}'s {:d} feature is add.".format(
                        chunk_index,
                        sample_index,
                    )
                )

            if is_first_dataset is True:
                dataset = data
                is_first_dataset = False
            else:
                dataset = np.dstack((dataset, data))

            print("***Subject {:d}'s dataset is added.".format(chunk_index))

        return dataset, labelset
