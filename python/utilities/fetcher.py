import glob

from dataclasses import dataclass
from typing import Optional


class Path:
    def __init__(
        self,
        _research_question: int,
        _body_part: str,
        _label: int,
        _authentication_flag: bool,
    ):
        self.research_question = _research_question
        self.body_part = _body_part
        self.label = _label
        self.authentication_flag = _authentication_flag

    def sort_paths(self) -> list:
        if self.authentication_flag is False:
            sorted_paths = sorted(
                glob.glob(
                    "./data/research_question"
                    + str(self.research_question)
                    + "/"
                    + self.body_part
                    + "_IMU_"
                    + str(self.label)
                    + "_*.txt"
                )
            )
        else:
            sorted_paths = sorted(
                glob.glob(
                    "./data/research_question"
                    + str(self.research_question)
                    + "/"
                    + self.body_part
                    + "_IMU_"
                    + "20"
                    + "_"
                    + str(self.label)
                    + "_*.txt"
                )
            )

        return sorted_paths


class Fetcher:
    def __init__(
        self,
        _research_question: int,
        _ages: list,
        _sample_length: int,
        _authentication_flag: bool,
        _authentication_classes: Optional[list] = None,
    ) -> None:
        self.research_question = _research_question
        self.ages = _ages
        self.sample_length = _sample_length
        self.authentication_flag = _authentication_flag
        self.authentication_classes = _authentication_classes

    def _parse_txt_data(self, txt_file_path: str) -> list:
        with open(txt_file_path, "r") as file:
            data = []
            buffer = []

            for index, row in enumerate(file):
                row_data = row.split()

                if (index > 0) and (index % self.sample_length == 0):
                    data.append(buffer)
                    buffer = []

                buffer.append(row_data)

        return data

    def fetch_dataset_chunks(self) -> zip:
        hand_data_chunk = []
        wrist_data_chunk = []
        helical_data_chunk = []
        labels_chunk = []

        if self.authentication_classes is not None:
            labels = self.authentication_classes
        else:
            labels = self.ages

        for label in labels:
            hand_paths = Path(
                self.research_question,
                "Hand",
                label,
                self.authentication_flag,
            ).sort_paths()
            wrist_paths = Path(
                self.research_question,
                "Wrist",
                label,
                self.authentication_flag,
            ).sort_paths()
            helical_paths = Path(
                self.research_question,
                "Helical",
                label,
                self.authentication_flag,
            ).sort_paths()

            paths = zip(hand_paths, wrist_paths, helical_paths)

            print("{:d}s data paths are imported".format(label))

            for hand_path, wrist_path, helical_path in paths:
                hand_data_txt = self._parse_txt_data(hand_path)
                wrist_data_txt = self._parse_txt_data(wrist_path)
                helical_data_txt = self._parse_txt_data(helical_path)

                hand_data_txt_length = len(hand_data_txt)
                wrist_data_txt_length = len(wrist_data_txt)
                helical_data_txt_length = len(helical_data_txt)

                data_txt_length = min(
                    hand_data_txt_length,
                    wrist_data_txt_length,
                    helical_data_txt_length,
                )

                if data_txt_length > 0:
                    hand_data_chunk.append(hand_data_txt)
                    wrist_data_chunk.append(wrist_data_txt)
                    helical_data_chunk.append(helical_data_txt)

                    labels_chunk.append(label)

        print("All data is imported.")
        print("Compressing all data to chunks.")

        dataset_chunks = zip(
            hand_data_chunk,
            wrist_data_chunk,
            helical_data_chunk,
            labels_chunk,
        )

        print("Data chunk(Hand, Wrist, Helical, and Labels) is created.")

        return dataset_chunks
