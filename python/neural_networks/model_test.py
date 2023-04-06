import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


loss_function = nn.BCELoss()


class ModelTest:
    def __init__(
        self, model: nn.Module, saving_path: str, training_status: bool
    ) -> None:
        self.model = model

        self.saving_path = saving_path
        self.training_status = training_status

    def test_step(self, model: nn.Module, data: torch.Tensor, labels: torch.Tensor):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=False)
        # loss = loss_object(labels, predictions)

        loss = loss_function(predictions, labels)

        return loss

    def get_trained_model(self):
        self.model.load_state_dict(torch.load(self.saving_path))

    def test_model(
        self,
        _test_data: torch.Tensor,
        _test_labels: torch.Tensor,
    ) -> None:
        if self.training_status is False:
            SystemError("Please train a model in advance.")

        test_dataset = DataLoader(TensorDataset(_test_data, _test_labels))

        for test_data, test_labels in test_dataset:
            test_labels_as_one_hot = F.one_hot(torch.squeeze(test_labels), 2)

            test_loss = self.test_step(self.model, test_data, test_labels_as_one_hot)

            print(
                f"Test Loss: {test_loss.item()}, "
                f"Test Accuracy: {(1 -  test_loss.item()) * 100}"
            )
