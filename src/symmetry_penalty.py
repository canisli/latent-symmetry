"""
Symmetry penalty for training with orbit variance regularization.

Penalty types:
- N_hPenalty: Numerator of Q_h = E[||h(g1*x) - h(g2*x)||²] (raw activations)
- N_zPenalty: Numerator of Q = E[||z(g1*x) - z(g2*x)||²] (PCA-projected)
- Q_hPenalty: Full Q_h = numerator / denominator (raw activations)
- Q_zPenalty: Full Q = numerator / denominator (PCA-projected)
- PeriodicPCAOrbitVariancePenalty: Periodic PCA re-fitting variant
- EMAPCAOrbitVariancePenalty: EMA statistics variant
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


class N_hPenalty(SymmetryPenalty):
    """
    Numerator of Q_h: E[||h(g1*x) - h(g2*x)||²] using raw activations.
    
    Computes only the orbit variance numerator, without normalization by data variance.
    """
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        data = data.to(device)
        N = data.shape[0]
        
        total_variance = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            diff_sq = ((h1 - h2) ** 2).sum(dim=-1).mean()
            total_variance = total_variance + diff_sq
        
        return total_variance / n_augmentations


class N_zPenalty(SymmetryPenalty):
    """
    Numerator of Q: E[||z(g1*x) - z(g2*x)||²] using PCA-projected activations.
    
    Mean and covariance computed fresh on each batch.
    Computes only the orbit variance numerator, without normalization by data variance.
    """
    
    def __init__(self, explained_variance: float = 0.95):
        self.explained_variance = explained_variance
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        data = data.to(device)
        N = data.shape[0]
        
        # Compute batch statistics (no grad)
        with torch.no_grad():
            h_batch = model.forward_with_intermediate(data, layer_idx)
            mu = h_batch.mean(dim=0)
            h_centered = h_batch - mu
            cov = (h_centered.T @ h_centered) / (h_batch.shape[0] - 1)
            
            if h_batch.shape[1] == 1:
                U = torch.eye(1, device=device)
            else:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                total_var = eigenvalues.sum()
                cumsum = torch.cumsum(eigenvalues, dim=0)
                k = (cumsum < self.explained_variance * total_var).sum().item() + 1
                k = max(1, min(k, len(eigenvalues)))
                U = eigenvectors[:, :k]
        
        total_variance = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            z1 = (h1 - mu) @ U
            z2 = (h2 - mu) @ U
            
            diff_sq = ((z1 - z2) ** 2).sum(dim=-1).mean()
            total_variance = total_variance + diff_sq
        
        return total_variance / n_augmentations


class Q_hPenalty(SymmetryPenalty):
    """
    Full Q_h metric: E[||h(g1*x) - h(g2*x)||²] / E[||h(x) - h(x')||²]
    
    Ratio of orbit variance to data variance using raw activations.
    Mean and denominator computed fresh on each batch.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: Small constant for numerical stability in division.
        """
        self.epsilon = epsilon
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        data = data.to(device)
        N = data.shape[0]
        
        # Compute denominator: E[||h(x) - h(x')||²] from batch
        with torch.no_grad():
            h_batch = model.forward_with_intermediate(data, layer_idx)
            h_diff = h_batch.unsqueeze(0) - h_batch.unsqueeze(1)  # (N, N, D)
            h_diff_sq = (h_diff ** 2).sum(dim=-1)  # (N, N)
            mask = ~torch.eye(N, dtype=torch.bool, device=device)
            denominator = h_diff_sq[mask].mean()
        
        # Compute numerator: E[||h(g1*x) - h(g2*x)||²]
        numerator = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            diff_sq = ((h1 - h2) ** 2).sum(dim=-1).mean()
            numerator = numerator + diff_sq
        
        numerator = numerator / n_augmentations
        
        return numerator / (denominator + self.epsilon)


class Q_zPenalty(SymmetryPenalty):
    """
    Full Q metric: E[||z(g1*x) - z(g2*x)||²] / E[||z(x) - z(x')||²]
    
    Ratio of orbit variance to data variance using PCA-projected activations.
    Mean, covariance, and denominator computed fresh on each batch.
    """
    
    def __init__(self, explained_variance: float = 0.95, epsilon: float = 1e-8):
        """
        Args:
            explained_variance: Fraction of variance to explain with PCA.
            epsilon: Small constant for numerical stability in division.
        """
        self.explained_variance = explained_variance
        self.epsilon = epsilon
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        data = data.to(device)
        N = data.shape[0]
        
        # Compute batch statistics and denominator (no grad)
        with torch.no_grad():
            h_batch = model.forward_with_intermediate(data, layer_idx)
            mu = h_batch.mean(dim=0)
            h_centered = h_batch - mu
            cov = (h_centered.T @ h_centered) / (h_batch.shape[0] - 1)
            
            if h_batch.shape[1] == 1:
                U = torch.eye(1, device=device)
            else:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                total_var = eigenvalues.sum()
                cumsum = torch.cumsum(eigenvalues, dim=0)
                k = (cumsum < self.explained_variance * total_var).sum().item() + 1
                k = max(1, min(k, len(eigenvalues)))
                U = eigenvectors[:, :k]
            
            # Compute z for batch and denominator
            z_batch = (h_batch - mu) @ U
            z_diff = z_batch.unsqueeze(0) - z_batch.unsqueeze(1)  # (N, N, k)
            z_diff_sq = (z_diff ** 2).sum(dim=-1)  # (N, N)
            mask = ~torch.eye(N, dtype=torch.bool, device=device)
            denominator = z_diff_sq[mask].mean()
        
        # Compute numerator: E[||z(g1*x) - z(g2*x)||²]
        numerator = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            z1 = (h1 - mu) @ U
            z2 = (h2 - mu) @ U
            
            diff_sq = ((z1 - z2) ** 2).sum(dim=-1).mean()
            numerator = numerator + diff_sq
        
        numerator = numerator / n_augmentations
        
        return numerator / (denominator + self.epsilon)


class PeriodicPCAOrbitVariancePenalty(SymmetryPenalty):
    """
    Orbit variance penalty with periodic PCA re-fitting.
    
    Re-fits the PCA projection every `refit_interval` calls to compute().
    This keeps the projection matrix aligned with the evolving model representations.
    """
    
    def __init__(self, explained_variance: float = 0.95, refit_interval: int = 100):
        """
        Args:
            explained_variance: Fraction of variance to explain with PCA (default 0.95).
            refit_interval: Re-fit PCA every N compute() calls (default 100).
        """
        self.explained_variance = explained_variance
        self.refit_interval = refit_interval
        # Per-layer projection parameters: {layer_idx: (mu, U)}
        self._projections = {}
        # Per-layer call counters: {layer_idx: count}
        self._call_counts = {}
        # Reference data for re-fitting (set via set_reference_data)
        self._reference_data = None
    
    def set_reference_data(self, data: torch.Tensor) -> None:
        """
        Set reference data used for periodic re-fitting.
        
        Args:
            data: Input data tensor of shape (N, 2) to use for PCA fitting.
        """
        self._reference_data = data
    
    def _fit_layer(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        """Fit PCA projection for a single layer."""
        model.eval()
        data = data.to(device)
        
        with torch.no_grad():
            h = model.forward_with_intermediate(data, layer_idx)
        
        # Compute mean and covariance
        mu = h.mean(dim=0)
        h_centered = h - mu
        cov = (h_centered.T @ h_centered) / (h.shape[0] - 1)
        
        # Get PCA projection
        if h.shape[1] == 1:
            U = torch.eye(1, device=device)
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            total_var = eigenvalues.sum()
            cumsum = torch.cumsum(eigenvalues, dim=0)
            k = (cumsum < self.explained_variance * total_var).sum().item() + 1
            k = max(1, min(k, len(eigenvalues)))
            U = eigenvectors[:, :k]
        
        self._projections[layer_idx] = (mu.detach(), U.detach())
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute PCA-projected orbit variance penalty, re-fitting periodically.
        """
        # Initialize or increment call counter
        if layer_idx not in self._call_counts:
            self._call_counts[layer_idx] = 0
        self._call_counts[layer_idx] += 1
        
        # Re-fit if needed (first call or every refit_interval)
        needs_refit = (
            layer_idx not in self._projections or
            self._call_counts[layer_idx] % self.refit_interval == 1
        )
        
        if needs_refit:
            # Use reference data if available, otherwise use current batch
            fit_data = self._reference_data if self._reference_data is not None else data
            self._fit_layer(model, fit_data, layer_idx, device)
        
        mu, U = self._projections[layer_idx]
        data = data.to(device)
        N = data.shape[0]
        
        total_variance = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            z1 = (h1 - mu) @ U
            z2 = (h2 - mu) @ U
            
            diff_sq = ((z1 - z2) ** 2).sum(dim=-1).mean()
            total_variance = total_variance + diff_sq
        
        return total_variance / n_augmentations


class EMAPCAOrbitVariancePenalty(SymmetryPenalty):
    """
    Orbit variance penalty with exponential moving average statistics for PCA.
    
    Maintains running EMA estimates of the mean and covariance matrix,
    updating them on each compute() call. This provides smooth adaptation
    to changing model representations without expensive periodic re-fitting.
    """
    
    def __init__(self, explained_variance: float = 0.95, ema_decay: float = 0.99):
        """
        Args:
            explained_variance: Fraction of variance to explain with PCA (default 0.95).
            ema_decay: EMA decay factor (default 0.99). Higher = slower adaptation.
        """
        self.explained_variance = explained_variance
        self.ema_decay = ema_decay
        # Per-layer EMA statistics: {layer_idx: (ema_mu, ema_cov, U)}
        self._stats = {}
    
    def _update_stats(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        """Update EMA statistics for a layer."""
        data = data.to(device)
        
        with torch.no_grad():
            h = model.forward_with_intermediate(data, layer_idx)
        
        # Compute batch statistics
        batch_mu = h.mean(dim=0)
        h_centered = h - batch_mu
        batch_cov = (h_centered.T @ h_centered) / (h.shape[0] - 1)
        
        if layer_idx not in self._stats:
            # Initialize with batch statistics
            ema_mu = batch_mu.detach()
            ema_cov = batch_cov.detach()
        else:
            # EMA update
            old_mu, old_cov, _ = self._stats[layer_idx]
            ema_mu = self.ema_decay * old_mu + (1 - self.ema_decay) * batch_mu.detach()
            ema_cov = self.ema_decay * old_cov + (1 - self.ema_decay) * batch_cov.detach()
        
        # Compute PCA projection from EMA covariance
        if h.shape[1] == 1:
            U = torch.eye(1, device=device)
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(ema_cov)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            total_var = eigenvalues.sum()
            cumsum = torch.cumsum(eigenvalues, dim=0)
            k = (cumsum < self.explained_variance * total_var).sum().item() + 1
            k = max(1, min(k, len(eigenvalues)))
            U = eigenvectors[:, :k]
        
        self._stats[layer_idx] = (ema_mu, ema_cov, U.detach())
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute PCA-projected orbit variance penalty with EMA statistics.
        
        Updates the EMA mean and covariance on each call.
        """
        # Update EMA statistics
        self._update_stats(model, data, layer_idx, device)
        
        ema_mu, _, U = self._stats[layer_idx]
        data = data.to(device)
        N = data.shape[0]
        
        total_variance = torch.tensor(0.0, device=device)
        
        for _ in range(n_augmentations):
            theta1 = sample_rotations(N, device=device)
            theta2 = sample_rotations(N, device=device)
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            # Use EMA mean for centering
            z1 = (h1 - ema_mu) @ U
            z2 = (h2 - ema_mu) @ U
            
            diff_sq = ((z1 - z2) ** 2).sum(dim=-1).mean()
            total_variance = total_variance + diff_sq
        
        return total_variance / n_augmentations


def create_symmetry_penalty(penalty_type: str, **kwargs) -> SymmetryPenalty:
    """
    Factory function to create a symmetry penalty.
    
    Args:
        penalty_type: One of:
            - "N_h": N_hPenalty (numerator only, raw activations)
            - "N_z": N_zPenalty (numerator only, PCA-projected)
            - "Q_h": Q_hPenalty (full ratio, raw activations)
            - "Q_z": Q_zPenalty (full ratio, PCA-projected)
            - "periodic_pca": PeriodicPCAOrbitVariancePenalty (re-fits every N steps)
            - "ema_pca": EMAPCAOrbitVariancePenalty (EMA statistics)
        **kwargs: Additional arguments passed to the penalty constructor.
    
    Returns:
        SymmetryPenalty instance.
    """
    if penalty_type == "N_h":
        return N_hPenalty()
    elif penalty_type == "N_z":
        return N_zPenalty(**kwargs)
    elif penalty_type == "Q_h":
        return Q_hPenalty(**kwargs)
    elif penalty_type == "Q_z":
        return Q_zPenalty(**kwargs)
    elif penalty_type == "periodic_pca":
        return PeriodicPCAOrbitVariancePenalty(**kwargs)
    elif penalty_type == "ema_pca":
        return EMAPCAOrbitVariancePenalty(**kwargs)
    else:
        raise ValueError(
            f"Unknown penalty type: {penalty_type}. "
            f"Use 'N_h', 'N_z', 'Q_h', 'Q_z', 'periodic_pca', or 'ema_pca'."
        )
