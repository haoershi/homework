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
        512,
        output_dim,
        3,
        torch.nn.ReLU(),
        torch.nn.init.xavier_normal_,
    )
