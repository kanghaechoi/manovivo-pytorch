import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class NetLearning:
    def __init__(
        self,
        neural_net,
        _model_weights_path: str,
        _loss_function: torch.Tensor = F.binary_cross_entropy,
    ) -> None:
        self.model_weights_path = _model_weights_path
        self.loss_function = _loss_function

        self.is_trained: bool = False

    def _train_step(
        self,
        _model,
        _data: torch.Tensor,
        _labels: torch.LongTensor,
        _optimizer,
    ):
        predictions = _model(_data)

        loss = self.loss_function(predictions, _labels.to(torch.float))

        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

        return loss

    def _test_step(
        self,
        _model,
        _data: torch.Tensor,
        _labels: torch.LongTensor,
    ):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        breakpoint()
        predictions = _model(_data)

        loss = self.loss_function(predictions, _labels.to(torch.float))

        return loss

    def train(
        self,
        model,
        data: torch.FloatTensor,
        labels: torch.LongTensor,
        optimizer,
        epochs: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch

            dataset = TensorDataset(data, labels)

            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )

            for batch_index, (batch_data, batch_labels) in enumerate(data_loader):
                one_hot_labels = F.one_hot(
                    torch.squeeze(batch_labels),
                    num_classes=2,
                )

                training_loss = self._train_step(
                    model,
                    batch_data,
                    one_hot_labels,
                    optimizer,
                )
                # loss += training_loss.item() * batch_data.size(0)
                # running_corrects += torch.sum(preds == labels.data)

                print(
                    f"Epoch: {epoch + 1}, "
                    f"Batch: {batch_index + 1}, "
                    f"Loss: {training_loss.item()}, "
                    f"Training Accuracy: {(1-training_loss.item()) * 100}"
                )

        self.is_trained = True

        torch.save(model.state_dict(), self.model_weights_path)

    def evaluate(
        self,
        neural_network,
        data: torch.FloatTensor,
        labels: torch.LongTensor,
    ):
        if self.is_trained is False:
            SystemError("Please train a model in advance.")

        neural_network.load_state_dict(torch.load(self.model_weights_path))
        # dataset = TensorDataset(_data.to(_device), _labels.to(_device))

        # data_loader = DataLoader(dataset)

        # for _, (data, labels) in enumerate(data_loader):
        one_hot_labels = F.one_hot(torch.squeeze(labels), 2)

        test_loss = self._test_step(neural_network, data, one_hot_labels)

        print(
            f"Test Loss: {test_loss.item()}, "
            f"Test Accuracy: {(1 -  test_loss.item()) * 100}"
        )
