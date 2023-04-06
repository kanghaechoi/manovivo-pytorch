import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

loss_function = nn.BCELoss()


def training_step(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    optimizer: optim.Optimizer,
):
    predictions = model(data)
    loss = loss_function(predictions, labels)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss


class ModelTraining:
    def __init__(
        self,
        _model: nn.Module,
        _optimizer: optim.Optimizer,
        _epochs: int,
        _batch_size: int,
        _saving_path: str,
    ) -> None:
        self.model = _model
        self.optimizer = _optimizer
        self.epochs = _epochs
        self.batch_size = _batch_size

        self.training_status: bool = False
        self.saving_path = _saving_path

    def train_model(
        self,
        _training_data: torch.Tensor,
        _training_labels: torch.Tensor,
    ) -> None:
        for epoch in range(self.epochs):
            # Reset the metrics at the start of the next epoch

            training_dataset = DataLoader(
                TensorDataset(_training_data, _training_labels),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )

            for mini_batch_index, (
                mini_batch_training_data,
                mini_batch_training_labels,
            ) in enumerate(training_dataset):
                mini_batch_training_labels_as_one_hot = F.one_hot(
                    torch.squeeze(mini_batch_training_labels),
                    num_classes=2,
                )
                training_loss = training_step(
                    self.model,
                    mini_batch_training_data,
                    mini_batch_training_labels_as_one_hot,
                    self.optimizer,
                )

                print(
                    f"Epoch {epoch + 1}, "
                    f"Mini Batch {mini_batch_index + 1}, "
                    f"Training Loss: {training_loss.item()}, "
                    f"Training Accuracy: {(1-training_loss.item()) * 100}"
                )

        self.training_status = True

    def save_trained_model(self) -> None:
        if self.training_status is False:
            SystemError("Please train a model in advance.")

        torch.save(self.model.state_dict(), self.saving_path)
