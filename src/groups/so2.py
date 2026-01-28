"""
SO(2) group action for 2D rotations.

SO(2) is the group of 2D rotations, parameterized by angle θ ∈ [0, 2π).
"""

import torch
import math


def rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate 2D points by angles theta.
    
    Args:
        x: Points of shape (N, 2) or (2,)
        theta: Rotation angles in radians, shape (N,) or scalar
    
    Returns:
        Rotated points with same shape as x
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    if x.dim() == 1:
        # Single point
        x_rot = torch.stack([
            cos_t * x[0] - sin_t * x[1],
            sin_t * x[0] + cos_t * x[1]
        ])
    else:
        # Batch of points: x is (N, 2), theta is (N,) or scalar
        if theta.dim() == 0:
            # Scalar theta - same rotation for all points
            x_rot = torch.stack([
                cos_t * x[:, 0] - sin_t * x[:, 1],
                sin_t * x[:, 0] + cos_t * x[:, 1]
            ], dim=1)
        else:
            # Per-point rotation
            x_rot = torch.stack([
                cos_t * x[:, 0] - sin_t * x[:, 1],
                sin_t * x[:, 0] + cos_t * x[:, 1]
            ], dim=1)
    
    return x_rot


def sample_rotations(n: int, device=None, generator=None) -> torch.Tensor:
    """
    Sample n random rotation angles uniformly from [0, 2*pi).
    
    Args:
        n: Number of angles to sample
        device: Torch device
        generator: Optional torch.Generator for reproducible sampling
    
    Returns:
        Tensor of shape (n,) with random angles
    """
    return torch.rand(n, device=device, generator=generator) * 2 * math.pi


def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """
    Construct 2x2 rotation matrix from angle.
    
    Args:
        theta: Rotation angle in radians (scalar or batch)
    
    Returns:
        Rotation matrix of shape (2, 2) or (N, 2, 2)
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    if theta.dim() == 0:
        return torch.tensor([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ], device=theta.device)
    else:
        # Batch: theta is (N,)
        N = theta.shape[0]
        R = torch.zeros(N, 2, 2, device=theta.device)
        R[:, 0, 0] = cos_t
        R[:, 0, 1] = -sin_t
        R[:, 1, 0] = sin_t
        R[:, 1, 1] = cos_t
        return R


def identity(device=None) -> torch.Tensor:
    """Return identity rotation (angle 0)."""
    return torch.tensor(0.0, device=device)
