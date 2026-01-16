import torch
import numpy as np

# TODO: Move featurization from utils.py to features.py
def fourier_coord_feature(x: torch.Tensor, B):
    x_proj = (2.0 * np.pi * x) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
