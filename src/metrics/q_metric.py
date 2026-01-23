"""
Q metric: Orbit variance ratio for measuring SO(2) invariance.

Q_l = E[||z(g1*x) - z(g2*x)||²] / E[||z(x) - z(x')||²]

- Q ≈ 0: Perfect invariance (orbit has no variance)
- Q ≈ 1: No invariance (orbit variance equals data variance)
- Q < 1: Partial invariance
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from pathlib import Path

from .base import BaseMetric
from .registry import register
from ..groups.so2 import rotate, sample_rotations


def compute_layer_statistics(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance of layer activations.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        device: Torch device.
    
    Returns:
        Tuple of (mu, cov) - mean and covariance of activations.
    """
    model.eval()
    model.to(device)
    data = data.to(device)
    
    with torch.no_grad():
        h = model.forward_with_intermediate(data, layer_idx)
    
    mu = h.mean(dim=0)
    h_centered = h - mu
    cov = (h_centered.T @ h_centered) / (h.shape[0] - 1)
    
    return mu, cov


def get_pca_projection(
    cov: torch.Tensor,
    explained_variance: float = 0.95,
) -> torch.Tensor:
    """
    Get top-k eigenvectors explaining specified variance.
    
    Args:
        cov: Covariance matrix of shape (D, D).
        explained_variance: Fraction of variance to explain (default 0.95).
    
    Returns:
        U: Projection matrix of shape (D, k) where k is chosen to explain variance.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Find k to explain specified variance
    total_var = eigenvalues.sum()
    cumsum = torch.cumsum(eigenvalues, dim=0)
    k = (cumsum < explained_variance * total_var).sum().item() + 1
    k = max(1, min(k, len(eigenvalues)))
    
    return eigenvectors[:, :k]


def project_activations(
    h: torch.Tensor,
    U: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """
    Project activations onto PCA subspace: z = U^T (h - mu).
    
    Args:
        h: Activations of shape (N, D).
        U: Projection matrix of shape (D, k).
        mu: Mean of shape (D,).
    
    Returns:
        Projected activations of shape (N, k).
    """
    return (h - mu) @ U


def compute_Q_h(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    n_rotations: int = 32,
    device: torch.device = None,
) -> float:
    """
    Compute orbit variance metric Q_h for a layer using raw activations (no PCA).
    
    Q_h_l = E[||h(g1*x) - h(g2*x)||^2] / E[||h(x) - h(x')||^2]
    
    Lower Q_h means more invariant to SO(2) transformations.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        n_rotations: Number of rotation pairs to sample per point.
        device: Torch device.
    
    Returns:
        Q_h value for the layer.
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    data = data.to(device)
    N = data.shape[0]
    
    # Get raw activations for original data
    with torch.no_grad():
        h = model.forward_with_intermediate(data, layer_idx)
    
    # Compute denominator: E[||h(x) - h(x')||^2]
    # Use all pairs: mean of ||h_i - h_j||^2 for i != j
    h_diff = h.unsqueeze(0) - h.unsqueeze(1)  # (N, N, D)
    h_diff_sq = (h_diff ** 2).sum(dim=-1)  # (N, N)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    denominator = h_diff_sq[mask].mean()
    
    # Compute numerator: E[||h(g1*x) - h(g2*x)||^2]
    numerator = 0.0
    for _ in range(n_rotations):
        theta1 = sample_rotations(N, device=device)
        theta2 = sample_rotations(N, device=device)
        
        x_rot1 = rotate(data, theta1)
        x_rot2 = rotate(data, theta2)
        
        with torch.no_grad():
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
        
        numerator += ((h1 - h2) ** 2).sum(dim=-1).mean()
    
    numerator /= n_rotations
    
    Q_h = (numerator / denominator).item()
    return Q_h


def compute_all_Q_h(
    model: nn.Module,
    data: torch.Tensor,
    n_rotations: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Compute Q_h for all layers in the model.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        n_rotations: Number of rotation pairs to sample per point.
        device: Torch device.
    
    Returns:
        Dictionary mapping layer names to Q_h values.
    """
    Q_h_values = {}
    
    # Hidden layers (1-indexed)
    for layer_idx in range(1, model.num_linear_layers):
        Q_h = compute_Q_h(model, data, layer_idx, n_rotations, device)
        Q_h_values[f'layer_{layer_idx}'] = Q_h
    
    # Output layer
    Q_h_out = compute_Q_h(model, data, -1, n_rotations, device)
    Q_h_values['output'] = Q_h_out
    
    return Q_h_values


def compute_Q(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    n_rotations: int = 32,
    explained_variance: float = 0.95,
    device: torch.device = None,
) -> float:
    """
    Compute orbit variance metric Q_l for a layer.
    
    Q_l = E[||z(g1*x) - z(g2*x)||^2] / E[||z(x) - z(x')||^2]
    
    Lower Q means more invariant to SO(2) transformations.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        n_rotations: Number of rotation pairs to sample per point.
        explained_variance: Fraction of variance for PCA (ignored for output layer).
        device: Torch device.
    
    Returns:
        Q value for the layer.
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    data = data.to(device)
    N = data.shape[0]
    
    # Compute layer statistics
    mu, cov = compute_layer_statistics(model, data, layer_idx, device)
    
    # Get PCA projection (skip for scalar output)
    with torch.no_grad():
        h_test = model.forward_with_intermediate(data[:1], layer_idx)
    
    if h_test.shape[1] == 1:
        # Scalar output - no PCA needed
        U = torch.eye(1, device=device)
    else:
        U = get_pca_projection(cov, explained_variance)
    
    # Compute z for original data
    with torch.no_grad():
        h = model.forward_with_intermediate(data, layer_idx)
    z = project_activations(h, U, mu)
    
    # Compute denominator: E[||z(x) - z(x')||^2]
    # Use all pairs: mean of ||z_i - z_j||^2 for i != j
    z_diff = z.unsqueeze(0) - z.unsqueeze(1)  # (N, N, k)
    z_diff_sq = (z_diff ** 2).sum(dim=-1)  # (N, N)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    denominator = z_diff_sq[mask].mean()
    
    # Compute numerator: E[||z(g1*x) - z(g2*x)||^2]
    numerator = 0.0
    for _ in range(n_rotations):
        theta1 = sample_rotations(N, device=device)
        theta2 = sample_rotations(N, device=device)
        
        x_rot1 = rotate(data, theta1)
        x_rot2 = rotate(data, theta2)
        
        with torch.no_grad():
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
        
        z1 = project_activations(h1, U, mu)
        z2 = project_activations(h2, U, mu)
        
        numerator += ((z1 - z2) ** 2).sum(dim=-1).mean()
    
    numerator /= n_rotations
    
    Q = (numerator / denominator).item()
    return Q


def compute_all_Q(
    model: nn.Module,
    data: torch.Tensor,
    n_rotations: int = 32,
    explained_variance: float = 0.95,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Compute Q for all layers in the model.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        n_rotations: Number of rotation pairs to sample per point.
        explained_variance: Fraction of variance for PCA.
        device: Torch device.
    
    Returns:
        Dictionary mapping layer names to Q values.
    """
    Q_values = {}
    
    # Hidden layers (1-indexed)
    for layer_idx in range(1, model.num_linear_layers):
        Q = compute_Q(model, data, layer_idx, n_rotations, explained_variance, device)
        Q_values[f'layer_{layer_idx}'] = Q
    
    # Output layer
    Q_out = compute_Q(model, data, -1, n_rotations, explained_variance, device)
    Q_values['output'] = Q_out
    
    return Q_values


def compute_oracle_Q(
    data: torch.Tensor,
    targets: torch.Tensor,
    scalar_field_fn,
    n_rotations: int = 32,
    device: torch.device = None,
) -> float:
    """
    Compute Q for the oracle (perfect predictor where ŷ = y).
    
    This evaluates what Q would be at the output if the model perfectly
    predicted the true labels. For invariant targets, oracle Q ≈ 0.
    For non-invariant targets, oracle Q ≈ 1.
    
    Args:
        data: Input data tensor of shape (N, 2).
        targets: True target values of shape (N, 1).
        scalar_field_fn: Function (x, y, r) -> target used to compute labels for rotated points.
        n_rotations: Number of rotation pairs to sample.
        device: Torch device.
    
    Returns:
        Oracle Q value.
    """
    import numpy as np
    
    if device is None:
        device = torch.device('cpu')
    
    data = data.to(device)
    targets = targets.to(device)
    N = data.shape[0]
    
    # Denominator: E[||y - y'||²] using all pairs
    y = targets
    y_diff = y.unsqueeze(0) - y.unsqueeze(1)  # (N, N, 1)
    y_diff_sq = (y_diff ** 2).sum(dim=-1)  # (N, N)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    denominator = y_diff_sq[mask].mean()
    
    if denominator < 1e-10:
        # All targets are the same, Q is undefined
        return 0.0
    
    # Numerator: E[||y(g1*x) - y(g2*x)||²]
    numerator = 0.0
    for _ in range(n_rotations):
        theta1 = sample_rotations(N, device=device)
        theta2 = sample_rotations(N, device=device)
        
        x_rot1 = rotate(data, theta1)
        x_rot2 = rotate(data, theta2)
        
        # Compute true labels for rotated points
        x1_np, y1_np = x_rot1[:, 0].cpu().numpy(), x_rot1[:, 1].cpu().numpy()
        x2_np, y2_np = x_rot2[:, 0].cpu().numpy(), x_rot2[:, 1].cpu().numpy()
        r1 = np.sqrt(x1_np**2 + y1_np**2)
        r2 = np.sqrt(x2_np**2 + y2_np**2)
        
        y_rot1 = torch.tensor(scalar_field_fn(x1_np, y1_np, r1), dtype=torch.float32, device=device).unsqueeze(1)
        y_rot2 = torch.tensor(scalar_field_fn(x2_np, y2_np, r2), dtype=torch.float32, device=device).unsqueeze(1)
        
        numerator += ((y_rot1 - y_rot2) ** 2).sum(dim=-1).mean()
    
    numerator /= n_rotations
    
    Q = (numerator / denominator).item()
    return Q


def plot_Q_vs_layer(Q_values: Dict[str, float], save_path: Path = None, oracle_Q: float = None, run_name: str = None):
    """
    Plot Q as a function of layer depth.
    
    Args:
        Q_values: Dictionary mapping layer names to Q values.
        save_path: Optional path to save the plot.
        oracle_Q: Optional oracle Q value to show as a bar next to output.
        run_name: Optional run name to include in the title.
    """
    layers = list(Q_values.keys())
    values = list(Q_values.values())
    
    # Add oracle bar if provided
    if oracle_Q is not None:
        layers = layers + ['oracle']
        values = values + [oracle_Q]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(layers))
    
    # Color bars: steelblue for model layers, green for oracle
    colors = ['steelblue'] * (len(layers) - 1) + ['green'] if oracle_Q is not None else ['steelblue'] * len(layers)
    
    # Left: linear scale
    axes[0].bar(x, values, color=colors, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Q (Orbit Variance)')
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_title('Linear')
    
    # Right: log scale
    eps = 1e-6
    log_values = [max(v, eps) for v in values]
    axes[1].bar(x, log_values, color=colors, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1].set_xlabel('Layer')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_title('Log')
    
    # Set overall title with run name if provided
    title = 'SO(2) Invariance Metric by Layer'
    if run_name:
        title = f'{run_name}: {title}'
    fig.suptitle(title)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_Q_h_vs_layer(Q_h_values: Dict[str, float], save_path: Path = None, run_name: str = None):
    """
    Plot Q_h (raw activation orbit variance) as a function of layer depth.
    
    Args:
        Q_h_values: Dictionary mapping layer names to Q_h values.
        save_path: Optional path to save the plot.
        run_name: Optional run name to include in the title.
    """
    layers = list(Q_h_values.keys())
    values = list(Q_h_values.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(layers))
    
    colors = ['coral'] * len(layers)
    
    # Left: linear scale
    axes[0].bar(x, values, color=colors, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Q_h (Raw Activation Orbit Variance)')
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_title('Linear')
    
    # Right: log scale
    eps = 1e-6
    log_values = [max(v, eps) for v in values]
    axes[1].bar(x, log_values, color=colors, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1].set_xlabel('Layer')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_title('Log')
    
    # Set overall title with run name if provided
    title = 'Q_h (Raw Activation) SO(2) Invariance by Layer'
    if run_name:
        title = f'{run_name}: {title}'
    fig.suptitle(title)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@register("Q")
class QMetric(BaseMetric):
    """
    Q metric for measuring SO(2) invariance.
    
    Q = E[||z(g1*x) - z(g2*x)||²] / E[||z(x) - z(x')||²]
    
    Parameters:
        n_rotations: Number of rotation pairs to sample (default: 32)
        explained_variance: PCA variance threshold (default: 0.95)
    """
    
    name = "Q"
    
    def __init__(self, n_rotations: int = 32, explained_variance: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.n_rotations = n_rotations
        self.explained_variance = explained_variance
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute Q for all layers."""
        return compute_all_Q(
            model, 
            data, 
            n_rotations=kwargs.get('n_rotations', self.n_rotations),
            explained_variance=kwargs.get('explained_variance', self.explained_variance),
            device=device
        )
    
    def plot(
        self,
        values: Dict[str, float],
        save_path: Path = None,
        **kwargs
    ) -> None:
        """Plot Q values with optional oracle reference line."""
        oracle_Q = kwargs.get('oracle_Q', None)
        run_name = kwargs.get('run_name', None)
        plot_Q_vs_layer(values, save_path, oracle_Q=oracle_Q, run_name=run_name)


@register("Q_h")
class Q_hMetric(BaseMetric):
    """
    Q_h metric for measuring SO(2) invariance using raw activations (no PCA).
    
    Q_h = E[||h(g1*x) - h(g2*x)||²] / E[||h(x) - h(x')||²]
    
    Unlike Q which uses PCA-projected activations, Q_h uses raw hidden layer
    activations directly.
    
    Parameters:
        n_rotations: Number of rotation pairs to sample (default: 32)
    """
    
    name = "Q_h"
    include_in_summary = False  # Not included in summary plots
    
    def __init__(self, n_rotations: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.n_rotations = n_rotations
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute Q_h for all layers."""
        return compute_all_Q_h(
            model, 
            data, 
            n_rotations=kwargs.get('n_rotations', self.n_rotations),
            device=device
        )
    
    def plot(
        self,
        values: Dict[str, float],
        save_path: Path = None,
        **kwargs
    ) -> None:
        """Plot Q_h values."""
        run_name = kwargs.get('run_name', None)
        plot_Q_h_vs_layer(values, save_path, run_name=run_name)
