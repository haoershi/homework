import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    A CNN for getting 0.55 accuracy on the CIFAR-10 dataset as quickly as possible.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Create a new model class.
        Arguments:
            num_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(16 * 10 * 10, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for CNN.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 10 * 10)
        x = self.fc1(x)
        return x
