"""
SL (Symmetry Loss) metric for measuring SO(2) invariance.

SL = (1/2) E[||h(g1*x) - h(g2*x)||²]

This is the unnormalized orbit variance - the raw expected squared distance
between representations of differently-rotated inputs.

- SL = 0: Perfect invariance
- SL > 0: Larger values indicate more variance under rotation
"""

import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path

from .base import BaseMetric
from .registry import register
from .plotting import plot_metric_vs_layer, TrainingInfo
from ..groups.so2 import rotate, sample_rotations


def compute_symmetry_loss(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    n_rotations: int = 32,
    device: torch.device = None,
) -> float:
    """
    Compute raw symmetry loss for a layer.
    
    SL = (1/2) E[||h(g1*x) - h(g2*x)||²]
    
    This is the unnormalized orbit variance - the raw expected squared distance
    between representations of differently-rotated inputs.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        n_rotations: Number of rotation pairs to sample per point.
        device: Torch device.
    
    Returns:
        SL value for the layer.
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    data = data.to(device)
    N = data.shape[0]
    
    # Compute SL: (1/2) E[||h(g1*x) - h(g2*x)||²]
    total_sl = 0.0
    for _ in range(n_rotations):
        theta1 = sample_rotations(N, device=device)
        theta2 = sample_rotations(N, device=device)
        
        x_rot1 = rotate(data, theta1)
        x_rot2 = rotate(data, theta2)
        
        with torch.no_grad():
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
        
        # ||h(g1*x) - h(g2*x)||² per sample
        diff_sq = ((h1 - h2) ** 2).sum(dim=-1)  # (N,)
        total_sl += diff_sq.mean()
    
    sl = (0.5 * total_sl / n_rotations).item()
    return sl


def compute_all_symmetry_loss(
    model: nn.Module,
    data: torch.Tensor,
    n_rotations: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Compute raw symmetry loss for all layers in the model.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        n_rotations: Number of rotation pairs to sample per point.
        device: Torch device.
    
    Returns:
        Dictionary mapping layer names to SL values.
    """
    sl_values = {}
    
    # Hidden layers (1-indexed)
    for layer_idx in range(1, model.num_linear_layers):
        sl = compute_symmetry_loss(model, data, layer_idx, n_rotations, device)
        sl_values[f'layer_{layer_idx}'] = sl
    
    # Output layer
    sl_out = compute_symmetry_loss(model, data, -1, n_rotations, device)
    sl_values['output'] = sl_out
    
    return sl_values


def plot_sl_vs_layer(
    sl_values: Dict[str, float],
    save_path: Path = None,
    run_name: str = None,
    sym_penalty_type: str = None,
    sym_layers: list = None,
    lambda_sym: float = 0.0,
):
    """
    Plot Symmetry Loss as a function of layer depth.
    
    Args:
        sl_values: Dictionary mapping layer names to SL values.
        save_path: Optional path to save the plot.
        run_name: Optional run name (unused, kept for API compatibility).
        sym_penalty_type: Type of symmetry penalty used during training.
        sym_layers: List of layers penalized during training.
        lambda_sym: Lambda value for symmetry penalty.
    """
    training_info = TrainingInfo(
        penalty_type=sym_penalty_type,
        layers=sym_layers,
        lambda_sym=lambda_sym,
    )
    plot_metric_vs_layer(
        values=sl_values,
        metric_name='SL',
        save_path=save_path,
        color='orchid',
        ylabel='SL (Symmetry Loss)',
        training_info=training_info,
        log_eps=1e-10,
    )


@register("SL")
class SLMetric(BaseMetric):
    """
    Raw Symmetry Loss metric for measuring SO(2) invariance.
    
    SL = (1/2) E[||h(g1*x) - h(g2*x)||²]
    
    This is the unnormalized orbit variance - the raw expected squared distance
    between representations of differently-rotated inputs. Unlike Q or RSL,
    this metric is not normalized and will scale with activation magnitudes.
    
    - SL = 0: Perfect invariance
    - SL > 0: Larger values indicate more variance under rotation
    
    Parameters:
        n_rotations: Number of rotation pairs to sample (default: 32)
    """
    
    name = "SL"
    
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
        """Compute SL for all layers."""
        return compute_all_symmetry_loss(
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
        """Plot SL values."""
        run_name = kwargs.get('run_name', None)
        plot_sl_vs_layer(values, save_path, run_name=run_name)
