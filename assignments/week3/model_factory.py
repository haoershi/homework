import torch
from model import MLP


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.

    Returns:
        MLP: The created model.

    """
    return MLP(
        input_dim,
        4096,
        output_dim,
        1,
        torch.nn.ReLU(),
        torch.nn.init.kaiming_normal_,
    )
