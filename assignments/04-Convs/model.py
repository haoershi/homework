import torch
import torch.nn as nn

# import torch.nn.functional as F
torch.manual_seed(12321)


class Model(torch.nn.Module):
    """
    A simple CNN model that reaches 0.55 accuracy that cost less than 15 sec for an epoch, with batch size 200
    and learning rate 6e-3 on CIFAR10 dataset
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Structure of the model that runs as fast as possible
        Args:
            num_channels (int): number of channels in the input image
            num_classes (int): number of classes in the dataset
        """
        super(Model, self).__init__()

        # self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ker = 5
        self.pad = (self.ker - 1) // 2
        self.nchan = 16
        self.conv2 = nn.Conv2d(
            num_channels,
            16,
            kernel_size=3,
            stride=2,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv1 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, num_classes)
        # self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        # self.fc2 = nn.Linear(256, 100)
        # self.fc3 = nn.Linear(100, num_classes)
        # nn.init.xavier_uniform_(self.conv2.weight)
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # self.convdepth = nn.Conv2d(num_channels,self.nchan,kernel_size = 1)
        # nn.init.xavier_uniform_(self.convdepth.weight)
        self.model = nn.Sequential(
            self.conv2,
            nn.ReLU(),
            self.bn2,
            self.pool2,
            # self.conv1,
            # nn.ReLU(),
            # self.bn2,
            # self.pool2,
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=6e-3)
        # criterion = nn.CrossEntropyLoss()
        # for _ in range(20):
        #     x = torch.randn(200, 3, 32, 32)
        #     y = torch.randn(200, 10)
        #     y_pred = self.model(x)
        #     loss = criterion(y_pred, y)
        #     loss.backward()
        #     # optimizer.step()
        #     optimizer.zero_grad()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): input image
        Returns:
            torch.Tensor: output logits
        """
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.pool1(x)

        # x = self.conv2(x)
        # # x = self.convdepth(x)
        # x = F.relu(x)
        # x = self.pool2(x)

        # x = x.view(-1, self.nchan * 8 * 8)
        # # x = F.relu(self.fc1(x))
        # # x = F.relu(self.fc2(x))
        # x = self.fc1(x)
        x = self.model(x)

        return x
