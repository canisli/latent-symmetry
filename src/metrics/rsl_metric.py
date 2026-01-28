"""
RSL (Relative Symmetry Loss) metric for measuring SO(2) invariance.

RSL = E[||h(g1*x) - h(g2*x)||² / (||h(g1*x)||² + ||h(g2*x)||² + ε)]

- RSL ≈ 0: Perfect invariance
- RSL ≈ 2: No invariance (orthogonal representations)
"""

import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path

from .base import BaseMetric
from .registry import register
from .plotting import plot_metric_vs_layer, TrainingInfo
from ..orbit import compute_relative_orbit_variance
from ..groups.so2 import rotate, sample_rotations


def compute_relative_symmetry_loss(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    n_rotations: int = 32,
    epsilon: float = 1e-8,
    device: torch.device = None,
) -> float:
    """
    Compute relative symmetry loss for a layer.
    
    RSL = E[||h(g1*x) - h(g2*x)||² / (||h(g1*x)||² + ||h(g2*x)||² + ε)]
    
    This is a scale-invariant measure of orbit variance. Unlike Q which normalizes
    by data variance, RSL normalizes each sample individually by the sum of norms,
    making it robust to varying activation scales across layers.
    
    - RSL ≈ 0: Perfect invariance (representations are identical under rotation)
    - RSL ≈ 2: No invariance (representations are uncorrelated/orthogonal)
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        n_rotations: Number of rotation pairs to sample per point.
        epsilon: Small constant for numerical stability.
        device: Torch device.
    
    Returns:
        RSL value for the layer.
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    data = data.to(device)
    
    # Use shared function for RSL computation
    rsl = compute_relative_orbit_variance(
        model, data, layer_idx, n_rotations, device,
        epsilon=epsilon, requires_grad=False
    )
    
    return rsl.item()


def compute_all_relative_symmetry_loss(
    model: nn.Module,
    data: torch.Tensor,
    n_rotations: int = 32,
    epsilon: float = 1e-8,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Compute relative symmetry loss for all layers in the model.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        n_rotations: Number of rotation pairs to sample per point.
        epsilon: Small constant for numerical stability.
        device: Torch device.
    
    Returns:
        Dictionary mapping layer names to RSL values.
    """
    rsl_values = {}
    
    # Hidden layers (1-indexed)
    for layer_idx in range(1, model.num_linear_layers):
        rsl = compute_relative_symmetry_loss(model, data, layer_idx, n_rotations, epsilon, device)
        rsl_values[f'layer_{layer_idx}'] = rsl
    
    # Output layer
    rsl_out = compute_relative_symmetry_loss(model, data, -1, n_rotations, epsilon, device)
    rsl_values['output'] = rsl_out
    
    return rsl_values


def compute_oracle_RSL(
    data: torch.Tensor,
    targets: torch.Tensor,
    scalar_field_fn,
    n_rotations: int = 32,
    epsilon: float = 1e-8,
    device: torch.device = None,
) -> float:
    """
    Compute RSL for the oracle (perfect predictor where ŷ = y).
    
    This evaluates what RSL would be at the output if the model perfectly
    predicted the true labels. For invariant targets, oracle RSL ≈ 0.
    For non-invariant targets, oracle RSL > 0.
    
    Args:
        data: Input data tensor of shape (N, 2).
        targets: True target values of shape (N, 1).
        scalar_field_fn: Function (x, y, r) -> target used to compute labels for rotated points.
        n_rotations: Number of rotation pairs to sample.
        epsilon: Small constant for numerical stability.
        device: Torch device.
    
    Returns:
        Oracle RSL value.
    """
    import numpy as np
    
    if device is None:
        device = torch.device('cpu')
    
    data = data.to(device)
    targets = targets.to(device)
    N = data.shape[0]
    
    total_rsl = 0.0
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
        
        # ||y(g1*x)||² + ||y(g2*x)||² per sample
        norm_sq_sum = (y_rot1 ** 2).sum(dim=-1) + (y_rot2 ** 2).sum(dim=-1)
        
        # Relative loss per sample
        rsl_per_sample = diff_sq / (norm_sq_sum + epsilon)
        
        total_rsl += rsl_per_sample.mean()
    
    rsl = (total_rsl / n_rotations).item()
    return rsl


def plot_rsl_vs_layer(
    rsl_values: Dict[str, float],
    save_path: Path = None,
    oracle_RSL: float = None,
    run_name: str = None,
    field_name: str = None,
    sym_penalty_type: str = None,
    sym_layers: list = None,
    lambda_sym: float = 0.0,
):
    """
    Plot Relative Symmetry Loss as a function of layer depth.
    
    Args:
        rsl_values: Dictionary mapping layer names to RSL values.
        save_path: Optional path to save the plot.
        oracle_RSL: Optional oracle RSL value to show as additional bar.
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
        values=rsl_values,
        metric_name='RSL',
        save_path=save_path,
        color='mediumseagreen',
        ylabel='RSL (Relative Symmetry Loss)',
        oracle_value=oracle_RSL,
        training_info=training_info,
    )


@register("RSL")
class RSLMetric(BaseMetric):
    """
    Relative Symmetry Loss metric for measuring SO(2) invariance.
    
    RSL = E[||h(g1*x) - h(g2*x)||² / (||h(g1*x)||² + ||h(g2*x)||² + ε)]
    
    This is a scale-invariant measure of orbit variance. It normalizes each 
    sample individually by the sum of representation norms, making it robust 
    to varying activation scales across layers.
    
    - RSL ≈ 0: Perfect invariance
    - RSL ≈ 2: No invariance (orthogonal representations)
    
    Parameters:
        n_rotations: Number of rotation pairs to sample (default: 32)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    name = "RSL"
    has_oracle = True
    
    def __init__(self, n_rotations: int = 32, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.n_rotations = n_rotations
        self.epsilon = epsilon
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute RSL for all layers."""
        return compute_all_relative_symmetry_loss(
            model, 
            data, 
            n_rotations=kwargs.get('n_rotations', self.n_rotations),
            epsilon=kwargs.get('epsilon', self.epsilon),
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
        """Compute oracle RSL value (perfect predictor)."""
        return compute_oracle_RSL(
            data, targets, scalar_field_fn,
            n_rotations=kwargs.get('n_rotations', self.n_rotations),
            epsilon=kwargs.get('epsilon', self.epsilon),
            device=device,
        )
    
    def plot(
        self,
        values: Dict[str, float],
        save_path: Path = None,
        oracle: float = None,
        **kwargs
    ) -> None:
        """Plot RSL values."""
        # Support both 'oracle' and legacy 'oracle_RSL' kwargs
        oracle_val = oracle if oracle is not None else kwargs.get('oracle_RSL', None)
        run_name = kwargs.get('run_name', None)
        plot_rsl_vs_layer(values, save_path, oracle_RSL=oracle_val, run_name=run_name)
