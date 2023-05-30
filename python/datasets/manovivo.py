import glob

import numpy as np


class Features:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate

    def read(self, feature_path: str) -> np.ndarray:
        with open(feature_path, "r") as f:
            features = []
            features_buffer = []

            sampling_counter = 1

            for row in f:
                feature = row.split()

                features_buffer.append(feature)

                if sampling_counter % self.sampling_rate == 0:
                    features.append(features_buffer)
                    features_buffer = []

                sampling_counter += 1

        features_as_array = np.array(features)

        return features_as_array

    def dstack(
        self,
        n_samples: int,
        features_1: np.ndarray,
        features_2: np.ndarray,
        features_3: np.ndarray,
    ) -> np.ndarray:
        features = np.dstack(
            (
                features_1[:n_samples, :, :],
                features_2[:n_samples, :, :],
                features_3[:n_samples, :, :],
            )
        )

        return features

    def concatenate(self, features_paths: dict):
        for features_path_index, features_path in enumerate(features_paths):
            hand_features = self.read(features_path["hand"])
            wrist_features = self.read(features_path["wrist"])
            helical_features = self.read(features_path["helical"])

            maximum_n_samples = min(
                hand_features.shape[0],
                wrist_features.shape[0],
                helical_features.shape[0],
            )

            features = self.dstack(
                maximum_n_samples,
                hand_features,
                wrist_features,
                helical_features,
            )

            if features_path_index == 0:
                dataset = features
            else:
                np.hstack((dataset, features))


class Manovivo(Features):
    def __init__(self, research_question: int, sampling_rate: int = 150) -> None:
        match research_question:
            case 1:
                self.research_profile = dict(
                    {
                        "research_question": "research_question_1",
                        "classes": ["20s", "50s", "70s"],
                    }
                )
            case 2:
                self.research_profile = dict(
                    {
                        "research_question": "research_question_2",
                        "classes": ["negative", "positive"],
                    }
                )
            case 3:
                self.research_profile = dict(
                    {
                        "research_question": "research_question_3",
                        "classes": ["negative", "positive"],
                    }
                )
            case _:
                raise ValueError()

        self.sampling_rate = sampling_rate
        self.body_parts = ["hand", "wrist", "helical"]

    def get_dataset(self, class_index: int) -> np.ndarray:
        datasets = []

        for body_part_index, body_part in enumerate(self.body_parts):
            path = sorted(
                glob.glob(
                    "./datasets/manovivo/"
                    + self.research_profile["research_question"]
                    + "/"
                    + self.research_profile["classes"][class_index]
                    + "/"
                    + body_part
                    + "/*.txt"
                )
            )

            if body_part_index == 0:
                hand_paths = path
                del path
            elif body_part_index == 1:
                wrist_paths = path
                del path
            elif body_part_index == 2:
                helical_paths = path
                del path

        features_paths = dict(
            {
                "hand": hand_paths,
                "wrist": wrist_paths,
                "helical": helical_paths,
            }
        )

        data = self.concatenate(features_paths)

        for features_path_index, features_path in enumerate(features_paths):
            breakpoint()
            hand_data = self.__read_features__(hand_features_path)
            wrist_data = self.__read_features__(wrist_features_path)
            helical_data = self.__read_features__(helical_features_path)

            number_of_samples = min(
                hand_data.shape[0],
                wrist_data.shape[0],
                helical_data.shape[0],
            )

            data = np.dstack(
                (
                    hand_data[:number_of_samples, :, :],
                    wrist_data[:number_of_samples, :, :],
                    helical_data[:number_of_samples, :, :],
                )
            )

            if features_paths == 0:
                dataset = data
            else:
                np.hstack((dataset, data))
