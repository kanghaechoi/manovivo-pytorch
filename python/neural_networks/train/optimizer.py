import torch.optim as optim


class Optimizer:
    def __init__(self, target: str, network_parameters, learning_rate: float) -> None:
        if target is "adam":
            self.optimzer = optim.Adam(params=network_parameters, lr=learning_rate)

        if target is "rmsprop":
            self.optimzer = optim.RMSprop(params=network_parameters, lr=learning_rate)

        return self.optimzer
