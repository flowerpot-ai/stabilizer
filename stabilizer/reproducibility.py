import os
import random
import numpy as np
import torch


def seed_torch(seed: int):
    """
    More information on Pytorch reproducibility can be found here: Pytorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    Args:
        seed (int): desired seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_everything(seed: int):
    """
    Call this function at the begining of your script to ensure reproducibility
    Args:
        seed (int): desired seed
    """
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    seed_torch()
