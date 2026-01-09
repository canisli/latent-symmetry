"""
easy, data sparse task
hard task like images (but the rotations are discrete)
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset


def compute_scalar_field(X, field_type='inv'):
    """
    Compute scalar field values for given 3D coordinates.
    
    Args:
        X: Tensor of shape (..., 3) containing 3D coordinates
        field_type: 'inv' for SO(3) invariant field, 'sph' for non-invariant spherical harmonic
    
    Returns:
        Tensor of shape (...,) containing scalar field values
    """
    if field_type == 'inv':
        # SO(3) invariant field: only depends on radius R
        R2 = X.square().sum(dim=-1, keepdim=True)
        result = torch.exp(-0.05 * R2) * torch.cos(2*R2)
        return result.squeeze(-1)
    elif field_type == 'sph':
        # Non-invariant spherical harmonic: Y_1^0 = z/r (breaks SO(3) symmetry)
        R = X.norm(dim=-1, keepdim=True)
        eps = 1e-8
        # Use z/r component (normalized spherical harmonic Y_1^0)
        z_over_r = X[..., 2:3] / (R + eps)
        # Add radial envelope to make it well-behaved
        R2 = X.square().sum(dim=-1, keepdim=True)
        envelope = torch.exp(-0.05 * R2)
        # NO constant offset - keeps relative symmetry loss meaningful
        result = (envelope * z_over_r * 5).squeeze(-1)
        return result
    else:
        raise ValueError(f"Unknown field_type: {field_type}. Must be 'inv' or 'sph'.")


class ScalarFieldDataset(Dataset):
    functional_form = "exp(-0.5 * R²) * cos(2 * R²)"
    
    def __init__(self, n_samples, seed=None, field_type='inv'):
        """
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            field_type: 'inv' for SO(3) invariant field, 'sph' for non-invariant spherical harmonic
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.field_type = field_type
        self.X = torch.rand((n_samples, 3)) * 10 - 5 # uniformly distributed in [-5, 5]
        y_flat = compute_scalar_field(self.X, field_type=field_type)
        # Ensure y has shape [n_samples, 1] to match model output shape
        self.y = y_flat.unsqueeze(-1) if y_flat.dim() == 1 else y_flat
        
        # Update functional_form based on field_type
        if field_type == 'inv':
            self.functional_form = "exp(-0.5 * R²) * cos(2 * R²)"
        elif field_type == 'sph':
            self.functional_form = "exp(-0.05 * R²) * (z/r) * 5"

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]