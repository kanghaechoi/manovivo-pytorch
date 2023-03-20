import glob


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
            paths = sorted(
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
            paths = sorted(
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

        return paths
