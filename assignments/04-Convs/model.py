import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """_summary_

        Args:
            num_channels (int): _description_
            num_classes (int): _description_
        """
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 16 * 8 * 8)
        x = self.fc1(x)

        return x
