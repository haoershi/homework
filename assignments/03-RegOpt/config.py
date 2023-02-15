from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    """_summary_"""

    batch_size = 32
    num_epochs = 10
    initial_learning_rate = 0.008
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        # 'lr_lambda': [lambda epoch: 0.999],
        # 'start_epoch': 20,
        # 'step_size': 100,
        # "gamma": 0.9,
        # "milestones": [3000, 3600, 4000, 4800, 5400, 6000],
        "T_0": 10,
        "eta_min": 0.0004,
        "T_mult": 3,
    }  # gamma=0.9, milestones=[30,80]

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )  # lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False

    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # RandomHorizontalFlip(),
            # RandomCrop(32, padding=4)
        ]
    )
