"""
easy, data sparse task
hard task like images (but the rotations are discrete)
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset


def compute_scalar_field(X):
    """
    Compute scalar field values for given 3D coordinates.
    
    Args:
        X: Tensor of shape (..., 3) containing 3D coordinates
    
    Returns:
        Tensor of shape (...,) containing scalar field values
    """
    R2 = X.square().sum(dim=-1, keepdim=True)
    return torch.exp(-0.5 * R2) * torch.cos(2*R2)


class ScalarFieldDataset(Dataset):
    def __init__(self, n_samples, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.X = torch.rand((n_samples, 3)) * 10 - 5 # uniformly distributed in [-5, 5]
        self.y = compute_scalar_field(self.X)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

