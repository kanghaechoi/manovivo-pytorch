import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader


class NetLearning:
    def __init__(
        self,
        _save_path: str,
        _loss_function: nn.Module = nn.BCELoss(),
    ) -> None:
        self.save_path = _save_path
        self.loss_function = _loss_function

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_trained: bool = False

    def _train_step(
        self,
        _model: nn.Module,
        _data: torch.Tensor,
        _labels: torch.Tensor,
    ):
        predictions = _model(_data)
        loss = self.loss_function(predictions, _labels)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss

    def _test_step(
        self,
        _model: nn.Module,
        _data: torch.Tensor,
        _labels: torch.Tensor,
    ):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = _model(_data)

        loss = self.loss_function(predictions, _labels)

        return loss

    def train(
        self,
        _model: nn.Module,
        _data: torch.Tensor,
        _labels: torch.Tensor,
        _epochs: int = 256,
        _batch_size: int = 64,
        _optimizer: optim.Optimizer = optim.Adam(),
        _shuffle: bool = True,
        _drop_last: bool = True,
    ) -> None:
        for epoch in range(_epochs):
            # Reset the metrics at the start of the next epoch

            dataset = (TensorDataset(_data.to(self.device), _labels.to(self.device)),)

            data_loader = DataLoader(
                dataset=dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                drop_last=_drop_last,
            )

            for batch_index, (batch_data, batch_labels) in enumerate(data_loader):
                one_hot_labels = F.one_hot(
                    torch.squeeze(batch_labels),
                    num_classes=2,
                )
                training_loss = self._train_step(
                    _model,
                    batch_data,
                    one_hot_labels,
                    _optimizer,
                )

                loss += training_loss.item() * batch_data.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print(
                    f"Epoch: {epoch + 1}, "
                    f"Mini Batch: {batch_index + 1}, "
                    f"Loss: {loss}, "
                    f"Training Accuracy: {(1-training_loss.item()) * 100}"
                )

        self.is_trained = True

        torch.save(_model.state_dict(), self.save_path)

    def test(
        self,
        _model: nn.Module,
        _data: torch.Tensor,
        _labels: torch.Tensor,
    ) -> None:
        if self.is_trained is False:
            SystemError("Please train a model in advance.")

        model = _model.load_state_dict(torch.load(self.save_path))

        dataset = TensorDataset(_data.to(self.device), _labels.to(self.device))

        data_loader = DataLoader(dataset)

        for data, labels in data_loader:
            one_hot_labels = F.one_hot(torch.squeeze(labels), 2)

            test_loss = self._test_step(model, data, one_hot_labels)

            print(
                f"Test Loss: {test_loss.item()}, "
                f"Test Accuracy: {(1 -  test_loss.item()) * 100}"
            )
