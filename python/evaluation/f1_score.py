import numpy as np


class F1ScoreBinary:
    def __init__(self, confusion_matrix: np.ndarray) -> None:
        confusion_matrix = confusion_matrix.reshape(-1)

        self.true_negative = confusion_matrix[0]
        self.false_positive = confusion_matrix[1]
        self.false_negative = confusion_matrix[2]
        self.true_positive = confusion_matrix[3]

        self.number_of_samples = np.sum(confusion_matrix)

    def accuracy(self) -> float:
        score = (
            (self.true_negative + self.true_positive) / self.number_of_samples
        ) * 100

        return score

    def recall(self) -> float:
        score = (self.true_positive / (self.true_positive + self.false_negative)) * 100

        return score

    def precision(self) -> float:
        score = (self.true_positive / (self.true_positive + self.false_positive)) * 100

        return score

    def specificity(self) -> float:
        score = (self.true_negative / (self.false_positive + self.true_negative)) * 100

        return score

    def f1_score(self) -> float:
        score = (
            (2 * self.precision + self.recall) / (self.precision + self.recall) * 100
        )

        return score


class F1Score3Dimension:
    def __init__(self, confusion_matrix: np.ndarray) -> None:
        confusion_matrix = confusion_matrix.reshape(-1)

        self.confusion_matrix_11 = confusion_matrix[0]
        self.confusion_matrix_12 = confusion_matrix[1]
        self.confusion_matrix_13 = confusion_matrix[2]

        self.confusion_matrix_21 = confusion_matrix[3]
        self.confusion_matrix_22 = confusion_matrix[4]
        self.confusion_matrix_23 = confusion_matrix[5]

        self.confusion_matrix_31 = confusion_matrix[6]
        self.confusion_matrix_32 = confusion_matrix[7]
        self.confusion_matrix_33 = confusion_matrix[8]

        self.number_of_samples = np.sum(confusion_matrix)

    def accuracy(self) -> float:
        score = (
            (self.true_negative + self.true_positive) / self.number_of_samples
        ) * 100

        return score

    def recall(self) -> float:
        score = (self.true_positive / (self.true_positive + self.false_negative)) * 100

        return score

    def precision(self) -> float:
        score = (self.true_positive / (self.true_positive + self.false_positive)) * 100

        return score

    def f1_score(self) -> float:
        score = (
            (2 * self.precision + self.recall) / (self.precision + self.recall) * 100
        )

        return score
