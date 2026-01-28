"""
Shared utilities for orbit variance computations.

This module provides common functions used by both:
- Symmetry penalties (training with gradients)
- Metrics (evaluation without gradients)
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional

from .groups.so2 import rotate, sample_rotations


# =============================================================================
# PCA Utilities
# =============================================================================

def pca_from_covariance(
    cov: torch.Tensor,
    explained_variance: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute PCA projection matrix from a covariance matrix.
    
    Args:
        cov: Covariance matrix of shape (D, D).
        explained_variance: Fraction of variance to explain (e.g., 0.95).
        device: Torch device.
    
    Returns:
        U: Projection matrix of shape (D, k) where k is chosen to explain
           the specified fraction of variance.
    """
    if cov.shape[0] == 1:
        return torch.eye(1, device=device)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    total_var = eigenvalues.sum()
    cumsum = torch.cumsum(eigenvalues, dim=0)
    k = (cumsum < explained_variance * total_var).sum().item() + 1
    k = max(1, min(k, len(eigenvalues)))
    
    return eigenvectors[:, :k]


def compute_pca_projection(
    h: torch.Tensor,
    explained_variance: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PCA projection matrix from activations.
    
    Args:
        h: Activation tensor of shape (N, D).
        explained_variance: Fraction of variance to explain (e.g., 0.95).
        device: Torch device.
    
    Returns:
        Tuple of (mu, U) where:
            mu: Mean vector of shape (D,).
            U: Projection matrix of shape (D, k) where k is chosen to explain
               the specified fraction of variance.
    """
    mu = h.mean(dim=0)
    h_centered = h - mu
    cov = (h_centered.T @ h_centered) / (h.shape[0] - 1)
    U = pca_from_covariance(cov, explained_variance, device)
    return mu, U


def project_activations(
    h: torch.Tensor,
    mu: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Project activations onto PCA subspace: z = (h - mu) @ U.
    
    Args:
        h: Activations of shape (N, D).
        mu: Mean of shape (D,).
        U: Projection matrix of shape (D, k).
    
    Returns:
        Projected activations of shape (N, k).
    """
    return (h - mu) @ U


# =============================================================================
# Variance Computations
# =============================================================================

def compute_pairwise_variance(representations: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise variance E[||f(x) - f(x')||²] for all pairs in a batch.
    
    Args:
        representations: Tensor of shape (N, D).
    
    Returns:
        Scalar tensor with the mean pairwise squared distance.
    """
    N = representations.shape[0]
    diff = representations.unsqueeze(0) - representations.unsqueeze(1)  # (N, N, D)
    diff_sq = (diff ** 2).sum(dim=-1)  # (N, N)
    mask = ~torch.eye(N, dtype=torch.bool, device=representations.device)
    return diff_sq[mask].mean()


def compute_orbit_variance(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    n_rotations: int,
    device: torch.device,
    generator: torch.Generator = None,
    transform_fn: Callable[[torch.Tensor], torch.Tensor] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    Compute orbit variance E[||f(g1*x) - f(g2*x)||²].
    
    This is the core computation shared between penalties and metrics.
    
    Args:
        model: Neural network model with forward_with_intermediate method.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        n_rotations: Number of rotation pairs to sample.
        device: Torch device.
        generator: Optional torch.Generator for reproducible rotation sampling.
        transform_fn: Optional function to transform activations (e.g., PCA projection).
                     If None, uses raw activations.
        requires_grad: If False, wraps computation in torch.no_grad() for evaluation.
    
    Returns:
        Scalar tensor with the orbit variance.
    """
    N = data.shape[0]
    total_variance = torch.tensor(0.0, device=device)
    
    context = torch.no_grad() if not requires_grad else _nullcontext()
    
    with context:
        for _ in range(n_rotations):
            theta1 = sample_rotations(N, device=device, generator=generator)
            theta2 = sample_rotations(N, device=device, generator=generator)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            if transform_fn is not None:
                h1 = transform_fn(h1)
                h2 = transform_fn(h2)
            
            diff_sq = ((h1 - h2) ** 2).sum(dim=-1).mean()
            total_variance = total_variance + diff_sq
    
    return total_variance / n_rotations


def compute_relative_orbit_variance(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    n_rotations: int,
    device: torch.device,
    epsilon: float = 1e-8,
    generator: torch.Generator = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    Compute relative orbit variance (RSL-style normalization).
    
    RSL = E[||h(g1*x) - h(g2*x)||² / (||h(g1*x)||² + ||h(g2*x)||² + ε)]
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index.
        n_rotations: Number of rotation pairs to sample.
        device: Torch device.
        epsilon: Small constant for numerical stability.
        generator: Optional torch.Generator for reproducible rotation sampling.
        requires_grad: If False, wraps computation in torch.no_grad().
    
    Returns:
        Scalar tensor with the relative orbit variance.
    """
    N = data.shape[0]
    total_rsl = torch.tensor(0.0, device=device)
    
    context = torch.no_grad() if not requires_grad else _nullcontext()
    
    with context:
        for _ in range(n_rotations):
            theta1 = sample_rotations(N, device=device, generator=generator)
            theta2 = sample_rotations(N, device=device, generator=generator)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            # ||h(g1*x) - h(g2*x)||² per sample
            diff_sq = ((h1 - h2) ** 2).sum(dim=-1)
            
            # ||h(g1*x)||² + ||h(g2*x)||² per sample
            norm_sq_sum = (h1 ** 2).sum(dim=-1) + (h2 ** 2).sum(dim=-1)
            
            # Relative loss per sample
            rsl_per_sample = diff_sq / (norm_sq_sum + epsilon)
            total_rsl = total_rsl + rsl_per_sample.mean()
    
    return total_rsl / n_rotations