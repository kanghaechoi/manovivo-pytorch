import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam()

training_loss = tf.keras.metrics.BinaryCrossentropy(name="train_loss")
training_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.BinaryCrossentropy(name="test_loss")
test_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_accuracy")


def training_step(
    model: nn.Module, data: torch.Tensor, labels: torch.Tensor, iterations: int
):
    for iteration in range(iterations):
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=True)
        loss = loss_function(predictions, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    training_loss(labels, predictions)
    training_accuracy(labels, predictions)


def test_step(model: nn.Module, data: torch.Tensor, labels: torch.Tensor):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(data, training=False)
    # loss = loss_object(labels, predictions)

    test_loss(labels, predictions)
    test_accuracy(labels, predictions)


class NNTraining:
    def __init__(
        self,
        _model: nn.Module,
        _optimizer: optim,
        _epochs: int,
        _batch_size: int,
        _saved_models_path: str,
    ) -> None:
        self.model = _model
        self.optimizer = _optimizer
        self.epochs = _epochs
        self.batch_size = _batch_size

        self.is_trained: bool = False
        self.saved_models_path = _saved_models_path

    def train_model(
        self,
        _training_data: np.ndarray,
        _training_labels: np.ndarray,
    ) -> None:
        for epoch in range(self.epochs):
            # Reset the metrics at the start of the next epoch
            training_loss.reset_states()
            training_accuracy.reset_states()

            training_dataset = (
                tf.data.Dataset.from_tensor_slices((_training_data, _training_labels))
                .shuffle(_training_data.shape[0] * 10)
                .batch(self.batch_size)
            )

            for small_training_data, small_training_labels in training_dataset:
                small_training_labels_as_one_hot = tf.one_hot(
                    tf.squeeze(small_training_labels),
                    2,
                )
                training_step(
                    self.model,
                    small_training_data,
                    small_training_labels_as_one_hot,
                )

            print(
                f"Epoch {epoch + 1}, "
                f"Training Loss: {training_loss.result()}, "
                f"Training Accuracy: {training_accuracy.result() * 100}"
            )

        self.is_trained = True

    def save_trained_model(self) -> None:
        if self.is_trained is False:
            SystemError("Please train a model in advance.")

        self.model.save(self.saved_models_path)

    def test_trained_model(
        self,
        _test_data: np.ndarray,
        _test_labels: np.ndarray,
    ) -> None:
        if self.is_trained is False:
            SystemError("Please train a model in advance.")

        test_loss.reset_states()
        test_accuracy.reset_states()

        test_dataset = tf.data.Dataset.from_tensors((_test_data, _test_labels))

        for test_data, test_labels in test_dataset:
            test_labels_as_one_hot = tf.one_hot(tf.squeeze(test_labels), 2)

            test_step(self.model, test_data, test_labels_as_one_hot)

        print(
            f"Test Loss: {test_loss.result()}, "
            f"Test Accuracy: {test_accuracy.result() * 100}"
        )
