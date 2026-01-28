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
from ..orbit import compute_orbit_variance
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
    
    # Compute orbit variance using shared function
    orbit_var = compute_orbit_variance(
        model, data, layer_idx, n_rotations, device, requires_grad=False
    )
    
    # SL = (1/2) * orbit_variance
    sl = (0.5 * orbit_var).item()
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


def compute_oracle_SL(
    data: torch.Tensor,
    targets: torch.Tensor,
    scalar_field_fn,
    n_rotations: int = 32,
    device: torch.device = None,
) -> float:
    """
    Compute SL for the oracle (perfect predictor where ŷ = y).
    
    This evaluates what SL would be at the output if the model perfectly
    predicted the true labels. For invariant targets, oracle SL ≈ 0.
    For non-invariant targets, oracle SL > 0.
    
    Args:
        data: Input data tensor of shape (N, 2).
        targets: True target values of shape (N, 1).
        scalar_field_fn: Function (x, y, r) -> target used to compute labels for rotated points.
        n_rotations: Number of rotation pairs to sample.
        device: Torch device.
    
    Returns:
        Oracle SL value.
    """
    import numpy as np
    
    if device is None:
        device = torch.device('cpu')
    
    data = data.to(device)
    targets = targets.to(device)
    N = data.shape[0]
    
    total_sl = 0.0
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
        
        # ||y(g1*x) - y(g2*x)||² per sample
        diff_sq = ((y_rot1 - y_rot2) ** 2).sum(dim=-1)
        total_sl += diff_sq.mean()
    
    sl = (0.5 * total_sl / n_rotations).item()
    return sl


def plot_sl_vs_layer(
    sl_values: Dict[str, float],
    save_path: Path = None,
    oracle_SL: float = None,
    run_name: str = None,
    field_name: str = None,
    sym_penalty_type: str = None,
    sym_layers: list = None,
    lambda_sym: float = 0.0,
):
    """
    Plot Symmetry Loss as a function of layer depth.
    
    Args:
        sl_values: Dictionary mapping layer names to SL values.
        save_path: Optional path to save the plot.
        oracle_SL: Optional oracle SL value to show as additional bar.
        run_name: Optional run name (unused, kept for API compatibility).
        field_name: Name of the scalar field used for training.
        sym_penalty_type: Type of symmetry penalty used during training.
        sym_layers: List of layers penalized during training.
        lambda_sym: Lambda value for symmetry penalty.
    """
    training_info = TrainingInfo(
        field_name=field_name,
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
        oracle_value=oracle_SL,
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
    has_oracle = True
    log_format = ".6f"  # SL values are typically small, need more precision
    
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
    
    def compute_oracle(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        scalar_field_fn,
        device: torch.device = None,
        **kwargs
    ) -> float:
        """Compute oracle SL value (perfect predictor)."""
        return compute_oracle_SL(
            data, targets, scalar_field_fn,
            n_rotations=kwargs.get('n_rotations', self.n_rotations),
            device=device,
        )
    
    def plot(
        self,
        values: Dict[str, float],
        save_path: Path = None,
        oracle: float = None,
        **kwargs
    ) -> None:
        """Plot SL values."""
        # Support both 'oracle' and legacy 'oracle_SL' kwargs
        oracle_val = oracle if oracle is not None else kwargs.get('oracle_SL', None)
        run_name = kwargs.get('run_name', None)
        plot_sl_vs_layer(values, save_path, oracle_SL=oracle_val, run_name=run_name)
