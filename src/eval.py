"""
Evaluation utilities and invariance metrics.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from pathlib import Path

from .so2 import rotate, sample_rotations


def plot_regression_surface(model, dataset, save_path, device):
    model.eval()
    model.to(device)
    
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(grid).cpu().numpy().reshape(xx.shape)
    
    true_radii = np.sqrt(xx**2 + yy**2)
    true_field = dataset.scalar_field_fn(xx, yy, true_radii)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin, vmax = true_field.min(), true_field.max()
    for ax, data, title in zip(axes, [preds, true_field, preds - true_field], 
                                ['Predicted', 'True', 'Error']):
        if title == 'Error':
            max_err = max(abs(data.min()), abs(data.max()), 0.1)
            cf = ax.contourf(xx, yy, data, levels=50, cmap='RdBu_r', vmin=-max_err, vmax=max_err)
            ax.contour(xx, yy, data, levels=[0], colors='black', linewidths=1)
        else:
            cf = ax.contourf(xx, yy, data, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.contour(xx, yy, data, levels=np.linspace(vmin, vmax, 11), colors='white', linewidths=0.5, alpha=0.7)
        plt.colorbar(cf, ax=ax)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')


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


def plot_Q_vs_layer(Q_values: Dict[str, float], save_path: Path = None):
    """
    Plot Q as a function of layer depth.
    
    Args:
        Q_values: Dictionary mapping layer names to Q values.
        save_path: Optional path to save the plot.
    """
    layers = list(Q_values.keys())
    values = list(Q_values.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(layers))
    ax.bar(x, values, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Q (Orbit Variance)')
    ax.set_title('SO(2) Invariance Metric by Layer')
    ax.set_ylim(bottom=0)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Q=1 (no invariance)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
