"""
Symmetry penalty for training with orbit variance regularization.

Penalty types:
- N_hPenalty: Numerator of Q_h = E[||h(g1*x) - h(g2*x)||²] (raw activations)
- N_zPenalty: Numerator of Q = E[||z(g1*x) - z(g2*x)||²] (PCA-projected)
- Q_hPenalty: Full Q_h = numerator / denominator (raw activations, stopgrad on denominator)
- Q_zPenalty: Full Q = numerator / denominator (PCA-projected, stopgrad on denominator)
- Q_h_ns: Q_h without stopgrad on denominator
- Q_z_ns: Q_z without stopgrad on denominator
- PeriodicPCAOrbitVariancePenalty: Periodic PCA re-fitting variant
- EMAPCAOrbitVariancePenalty: EMA statistics variant
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Callable

from .orbit import (
    compute_pca_projection,
    pca_from_covariance,
    compute_pairwise_variance,
    compute_orbit_variance,
)


# =============================================================================
# Base Class
# =============================================================================

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
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Compute the symmetry penalty for a single layer.
        
        Args:
            model: Neural network model.
            data: Input data tensor of shape (N, 2).
            layer_idx: Layer index (1-based for hidden, -1 for output).
            n_augmentations: Number of rotation pairs to sample.
            device: Torch device.
            generator: Optional torch.Generator for reproducible rotation sampling.
        
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
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Compute the total symmetry penalty summed over multiple layers.
        
        Args:
            model: Neural network model.
            data: Input data tensor of shape (N, 2).
            layer_indices: List of layer indices to penalize.
            n_augmentations: Number of rotation pairs to sample.
            device: Torch device.
            generator: Optional torch.Generator for reproducible rotation sampling.
        
        Returns:
            Scalar tensor with the total penalty value (with gradients).
        """
        if not layer_indices:
            return torch.tensor(0.0, device=device)
        
        total = torch.tensor(0.0, device=device)
        for layer_idx in layer_indices:
            total = total + self.compute(model, data, layer_idx, n_augmentations, device, generator)
        return total


# =============================================================================
# Penalty Implementations
# =============================================================================

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
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        data = data.to(device)
        return compute_orbit_variance(
            model, data, layer_idx, n_augmentations, device,
            generator=generator, requires_grad=True
        )


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
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        data = data.to(device)
        
        # Compute batch PCA (no grad for eigendecomposition)
        with torch.no_grad():
            h_batch = model.forward_with_intermediate(data, layer_idx)
            mu, U = compute_pca_projection(h_batch, self.explained_variance, device)
        
        # Define PCA transform
        def pca_transform(h):
            return (h - mu) @ U
        
        return compute_orbit_variance(
            model, data, layer_idx, n_augmentations, device,
            generator=generator, transform_fn=pca_transform, requires_grad=True
        )


class Q_hPenalty(SymmetryPenalty):
    """
    Full Q_h metric: E[||h(g1*x) - h(g2*x)||²] / E[||h(x) - h(x')||²]
    
    Ratio of orbit variance to data variance using raw activations.
    Mean and denominator computed fresh on each batch.
    """
    
    def __init__(self, epsilon: float = 1e-8, stopgrad_denominator: bool = True):
        """
        Args:
            epsilon: Small constant for numerical stability in division.
            stopgrad_denominator: If True, stop gradient through denominator to prevent
                the optimizer from inflating data variance. Default True.
        """
        self.epsilon = epsilon
        self.stopgrad_denominator = stopgrad_denominator
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        data = data.to(device)
        
        # Compute denominator: E[||h(x) - h(x')||²] from batch
        def get_denominator():
            h_batch = model.forward_with_intermediate(data, layer_idx)
            return compute_pairwise_variance(h_batch)
        
        if self.stopgrad_denominator:
            with torch.no_grad():
                denominator = get_denominator()
        else:
            denominator = get_denominator()
        
        # Compute numerator
        numerator = compute_orbit_variance(
            model, data, layer_idx, n_augmentations, device,
            generator=generator, requires_grad=True
        )
        
        return numerator / (denominator + self.epsilon)


class Q_zPenalty(SymmetryPenalty):
    """
    Full Q metric: E[||z(g1*x) - z(g2*x)||²] / E[||z(x) - z(x')||²]
    
    Ratio of orbit variance to data variance using PCA-projected activations.
    Mean, covariance, and denominator computed fresh on each batch.
    """
    
    def __init__(self, explained_variance: float = 0.95, epsilon: float = 1e-8, stopgrad_denominator: bool = True):
        """
        Args:
            explained_variance: Fraction of variance to explain with PCA.
            epsilon: Small constant for numerical stability in division.
            stopgrad_denominator: If True, stop gradient through denominator to prevent
                the optimizer from inflating data variance. Default True.
        """
        self.explained_variance = explained_variance
        self.epsilon = epsilon
        self.stopgrad_denominator = stopgrad_denominator
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        data = data.to(device)
        
        # Compute PCA statistics (always no grad - eigendecomposition is expensive/unstable)
        with torch.no_grad():
            h_batch = model.forward_with_intermediate(data, layer_idx)
            mu, U = compute_pca_projection(h_batch, self.explained_variance, device)
        
        # Define PCA transform
        def pca_transform(h):
            return (h - mu) @ U
        
        # Compute denominator: E[||z(x) - z(x')||²]
        def get_denominator():
            h_batch_denom = model.forward_with_intermediate(data, layer_idx)
            z_batch = pca_transform(h_batch_denom)
            return compute_pairwise_variance(z_batch)
        
        if self.stopgrad_denominator:
            with torch.no_grad():
                denominator = get_denominator()
        else:
            denominator = get_denominator()
        
        # Compute numerator
        numerator = compute_orbit_variance(
            model, data, layer_idx, n_augmentations, device,
            generator=generator, transform_fn=pca_transform, requires_grad=True
        )
        
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
            mu, U = compute_pca_projection(h, self.explained_variance, device)
        
        self._projections[layer_idx] = (mu.detach(), U.detach())
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
        generator: torch.Generator = None,
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
        
        # Define PCA transform using stored projection
        def pca_transform(h):
            return (h - mu) @ U
        
        return compute_orbit_variance(
            model, data, layer_idx, n_augmentations, device,
            generator=generator, transform_fn=pca_transform, requires_grad=True
        )


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
        U = pca_from_covariance(ema_cov, self.explained_variance, device)
        
        self._stats[layer_idx] = (ema_mu, ema_cov, U.detach())
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer_idx: int,
        n_augmentations: int,
        device: torch.device,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Compute PCA-projected orbit variance penalty with EMA statistics.
        
        Updates the EMA mean and covariance on each call.
        """
        # Update EMA statistics
        self._update_stats(model, data, layer_idx, device)
        
        ema_mu, _, U = self._stats[layer_idx]
        data = data.to(device)
        
        # Define PCA transform using EMA statistics
        def pca_transform(h):
            return (h - ema_mu) @ U
        
        return compute_orbit_variance(
            model, data, layer_idx, n_augmentations, device,
            generator=generator, transform_fn=pca_transform, requires_grad=True
        )


# =============================================================================
# Factory Function
# =============================================================================

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
            - stopgrad_denominator: For Q_h/Q_z, whether to stop gradient through denominator (default True).
            - explained_variance: For PCA-based penalties, fraction of variance to explain (default 0.95).
            - epsilon: For Q_h/Q_z, numerical stability constant (default 1e-8).
            - refit_interval: For periodic_pca, steps between re-fits (default 100).
            - ema_decay: For ema_pca, EMA decay factor (default 0.99).
    
    Returns:
        SymmetryPenalty instance.
    """
    # Filter kwargs based on what each penalty type accepts
    stopgrad_denominator = kwargs.pop('stopgrad_denominator', True)
    
    if penalty_type == "N_h":
        return N_hPenalty()
    elif penalty_type == "N_z":
        # N_z only accepts explained_variance
        filtered = {k: v for k, v in kwargs.items() if k in ['explained_variance']}
        return N_zPenalty(**filtered)
    elif penalty_type == "Q_h":
        # Q_h accepts epsilon, stopgrad_denominator
        filtered = {k: v for k, v in kwargs.items() if k in ['epsilon']}
        return Q_hPenalty(stopgrad_denominator=stopgrad_denominator, **filtered)
    elif penalty_type == "Q_z":
        # Q_z accepts explained_variance, epsilon, stopgrad_denominator
        filtered = {k: v for k, v in kwargs.items() if k in ['explained_variance', 'epsilon']}
        return Q_zPenalty(stopgrad_denominator=stopgrad_denominator, **filtered)
    elif penalty_type == "Q_h_ns":
        # Q_h with no stopgrad on denominator
        filtered = {k: v for k, v in kwargs.items() if k in ['epsilon']}
        return Q_hPenalty(stopgrad_denominator=False, **filtered)
    elif penalty_type == "Q_z_ns":
        # Q_z with no stopgrad on denominator
        filtered = {k: v for k, v in kwargs.items() if k in ['explained_variance', 'epsilon']}
        return Q_zPenalty(stopgrad_denominator=False, **filtered)
    elif penalty_type == "periodic_pca":
        # periodic_pca accepts explained_variance, refit_interval
        filtered = {k: v for k, v in kwargs.items() if k in ['explained_variance', 'refit_interval']}
        return PeriodicPCAOrbitVariancePenalty(**filtered)
    elif penalty_type == "ema_pca":
        # ema_pca accepts explained_variance, ema_decay
        filtered = {k: v for k, v in kwargs.items() if k in ['explained_variance', 'ema_decay']}
        return EMAPCAOrbitVariancePenalty(**filtered)
    else:
        raise ValueError(
            f"Unknown penalty type: {penalty_type}. "
            f"Use 'N_h', 'N_z', 'Q_h', 'Q_z', 'Q_h_ns', 'Q_z_ns', 'periodic_pca', or 'ema_pca'."
        )
