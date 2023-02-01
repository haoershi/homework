from typing import Callable
import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        # Assign activation function
        self.actv = activation
        self.input_size = input_size
        self.out_size = num_classes
        self.init = initializer

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # Initialize layers of MLP
        # self.layers = nn.ModuleList()

        # Loop over layers and create each one
        # for _ in range(hidden_count):
        #     self.layers += [nn.Linear(input_size, hidden_size)]
        #     input_size = hidden_size

        # self.out = nn.Linear(hidden_size, num_classes)

        for layer in self.layers:
            self.init(layer.weight)

        self.init(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # for layer in self.layers:
        #     x = self.actv(layer(x))
        out = self.fc1(x)
        out = self.actv(out)
        out = self.fc2(out)

        # x = self.out(x)

        return x
