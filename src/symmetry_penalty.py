"""
Symmetry penalty for training with orbit variance regularization.

Provides penalties based on the numerator of Q:
- RawOrbitVariancePenalty: E[||h(g1*x) - h(g2*x)||²] using raw activations
- PCAOrbitVariancePenalty: E[||z(g1*x) - z(g2*x)||²] using PCA-projected activations
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional

from .groups.so2 import rotate, sample_rotations


class SymmetryPenalty(ABC):
    """Abstract base class for symmetry penalties."""
    
    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the symmetry penalty for a single layer.
        
        Args:
            model: Neural network model.
            data: Input data tensor of shape (N, 2).
            layer_idx: Layer index (1-based for hidden, -1 for output).
            n_augmentations: Number of rotation pairs to sample.
            device: Torch device.
        
        Returns:
            Scalar tensor with the penalty value (with gradients).
        """
        pass
    
    def compute_total(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_indices: List[int],
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the total symmetry penalty summed over multiple layers.
        
        Args:
            model: Neural network model.
            data: Input data tensor of shape (N, 2).
            layer_indices: List of layer indices to penalize.
            n_augmentations: Number of rotation pairs to sample.
            device: Torch device.
        
        Returns:
            Scalar tensor with the total penalty value (with gradients).
        """
        if not layer_indices:
            return torch.tensor(0.0, device=device)
        
        total = torch.tensor(0.0, device=device)
        for layer_idx in layer_indices:
            total = total + self.compute(model, data, layer_idx, n_augmentations, device)
        return total


class RawOrbitVariancePenalty(SymmetryPenalty):
    """
    Orbit variance penalty using raw activations (Q_h numerator).
    
    Computes E[||h(g1*x) - h(g2*x)||²] where h is the raw layer activation
    and g1, g2 are random SO(2) rotations.
    """
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute raw orbit variance penalty for a layer.
        
        Note: Does NOT use torch.no_grad() so gradients flow for backprop.
        """
        data = data.to(device)
        N = data.shape[0]
        
        total_variance = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            # Forward pass WITH gradients
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            # ||h(g1*x) - h(g2*x)||² per sample, then mean
            diff_sq = ((h1 - h2) ** 2).sum(dim=-1).mean()
            total_variance = total_variance + diff_sq
        
        return total_variance / n_augmentations


class PCAOrbitVariancePenalty(SymmetryPenalty):
    """
    Orbit variance penalty using PCA-projected activations (Q numerator).
    
    Computes E[||z(g1*x) - z(g2*x)||²] where z = U^T(h - mu) is the 
    PCA-projected activation and g1, g2 are random SO(2) rotations.
    
    Requires calling fit() before use to compute the projection matrix.
    """
    
    def __init__(self, explained_variance: float = 0.95):
        """
        Args:
            explained_variance: Fraction of variance to explain with PCA (default 0.95).
        """
        self.explained_variance = explained_variance
        # Per-layer projection parameters: {layer_idx: (mu, U)}
        self._projections = {}
    
    def fit(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        """
        Compute PCA projection matrix for a layer.
        
        Args:
            model: Neural network model.
            data: Input data tensor of shape (N, 2).
            layer_idx: Layer index to fit.
            device: Torch device.
        """
        model.eval()
        model.to(device)
        data = data.to(device)
        
        with torch.no_grad():
            h = model.forward_with_intermediate(data, layer_idx)
        
        # Compute mean and covariance
        mu = h.mean(dim=0)
        h_centered = h - mu
        cov = (h_centered.T @ h_centered) / (h.shape[0] - 1)
        
        # Get PCA projection
        if h.shape[1] == 1:
            # Scalar output - no PCA needed
            U = torch.eye(1, device=device)
        else:
            U = self._get_pca_projection(cov)
        
        self._projections[layer_idx] = (mu.detach(), U.detach())
    
    def _get_pca_projection(self, cov: torch.Tensor) -> torch.Tensor:
        """Get top-k eigenvectors explaining specified variance."""
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Find k to explain specified variance
        total_var = eigenvalues.sum()
        cumsum = torch.cumsum(eigenvalues, dim=0)
        k = (cumsum < self.explained_variance * total_var).sum().item() + 1
        k = max(1, min(k, len(eigenvalues)))
        
        return eigenvectors[:, :k]
    
    def fit_all(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_indices: List[int],
        device: torch.device,
    ) -> None:
        """
        Fit PCA projections for multiple layers.
        
        Args:
            model: Neural network model.
            data: Input data tensor of shape (N, 2).
            layer_indices: List of layer indices to fit.
            device: Torch device.
        """
        for layer_idx in layer_indices:
            self.fit(model, data, layer_idx, device)
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute PCA-projected orbit variance penalty for a layer.
        
        Requires fit() to have been called for this layer first.
        
        Note: Does NOT use torch.no_grad() so gradients flow for backprop.
        """
        if layer_idx not in self._projections:
            raise ValueError(
                f"Layer {layer_idx} not fitted. Call fit() or fit_all() first."
            )
        
        mu, U = self._projections[layer_idx]
        data = data.to(device)
        N = data.shape[0]
        
        total_variance = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            # Forward pass WITH gradients
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            # Project onto PCA subspace: z = (h - mu) @ U
            z1 = (h1 - mu) @ U
            z2 = (h2 - mu) @ U
            
            # ||z(g1*x) - z(g2*x)||² per sample, then mean
            diff_sq = ((z1 - z2) ** 2).sum(dim=-1).mean()
            total_variance = total_variance + diff_sq
        
        return total_variance / n_augmentations


def create_symmetry_penalty(penalty_type: str, **kwargs) -> SymmetryPenalty:
    """
    Factory function to create a symmetry penalty.
    
    Args:
        penalty_type: "raw" for RawOrbitVariancePenalty, "pca" for PCAOrbitVariancePenalty.
        **kwargs: Additional arguments passed to the penalty constructor.
    
    Returns:
        SymmetryPenalty instance.
    """
    if penalty_type == "raw":
        return RawOrbitVariancePenalty()
    elif penalty_type == "pca":
        return PCAOrbitVariancePenalty(**kwargs)
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}. Use 'raw' or 'pca'.")
