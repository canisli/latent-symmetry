"""
Evaluation utilities and invariance metrics.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from pathlib import Path


def plot_regression_surface(model, dataset, save_path, device):
    model.eval()
    model.to(device)
    
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(grid).cpu().numpy().reshape(xx.shape)
    
    true_radii = np.sqrt(xx**2 + yy**2)
    true_field = dataset.scalar_field_fn(true_radii)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, data, title in zip(axes, [preds, true_field, preds - true_field], 
                                ['Predicted', 'True', 'Error']):
        if title == 'Error':
            max_err = max(abs(data.min()), abs(data.max()), 0.1)
            cf = ax.contourf(xx, yy, data, levels=50, cmap='RdBu_r', vmin=-max_err, vmax=max_err)
            ax.contour(xx, yy, data, levels=[0], colors='black', linewidths=1)
        else:
            cf = ax.contourf(xx, yy, data, levels=50, cmap='viridis', vmin=0, vmax=1)
            ax.contour(xx, yy, data, levels=np.linspace(0, 1, 11), colors='white', linewidths=0.5, alpha=0.7)
        plt.colorbar(cf, ax=ax)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')


def compute_invariance_metrics(
    model: nn.Module,
    data: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute invariance metrics for the model.
    
    To be implemented with specific symmetry analysis.
    
    Args:
        model: Neural network model.
        data: Input data tensor.
    
    Returns:
        Dictionary of invariance metrics.
    """
    # Placeholder for future implementation
    pass
