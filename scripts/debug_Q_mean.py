#!/usr/bin/env python
"""
Debug script to investigate Q metric behavior and mean effects.

The Q metric formula assumes:
  Q = E_x[Var_g(h(gx))] / Var_x(h)

Identity (when E[h] = 0):
  Q = 1 - E_x[||E_g h(gx)||²] / E_x[||h||²]

General formula (without zero-mean assumption):
  Let A = E[||h||²], B = ||E[h]||², C = E[||E_g h(gx)||²]
  Then: Q = (A - C) / (A - B)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import MLP
from groups.so2 import rotate, sample_rotations


def compute_Q_diagnostic(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    n_rotations: int = 64,
    n_orbit_samples: int = 64,
    device: torch.device = None,
):
    """
    Compute Q metric using multiple methods and report diagnostic statistics.
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    data = data.to(device)
    N = data.shape[0]
    
    # Get raw activations
    with torch.no_grad():
        h = model.forward_with_intermediate(data, layer_idx)
    
    D = h.shape[1]  # dimension of activations
    
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} Diagnostics (N={N}, D={D})")
    print(f"{'='*60}")
    
    # ==== Basic statistics ====
    h_mean = h.mean(dim=0)  # E[h], shape (D,)
    h_sq_norm_mean = (h ** 2).sum(dim=-1).mean()  # E[||h||²], scalar
    mean_sq_norm = (h_mean ** 2).sum()  # ||E[h]||², scalar
    
    # Variance via pairwise differences
    h_diff = h.unsqueeze(0) - h.unsqueeze(1)  # (N, N, D)
    h_diff_sq = (h_diff ** 2).sum(dim=-1)  # (N, N)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    pairwise_var = h_diff_sq[mask].mean() / 2  # Var(h) via pairwise
    
    # Variance via standard formula
    standard_var = h_sq_norm_mean - mean_sq_norm  # E[||h||²] - ||E[h]||²
    
    print(f"\nActivation statistics:")
    print(f"  E[||h||²] (A)      = {h_sq_norm_mean.item():.6f}")
    print(f"  ||E[h]||² (B)      = {mean_sq_norm.item():.6f}")
    print(f"  Var(h) = A - B     = {standard_var.item():.6f}")
    print(f"  Var(h) via pairs   = {pairwise_var.item():.6f}")
    print(f"  Mean activation    = {h_mean.mean().item():.6f} (avg over dims)")
    print(f"  Min activation     = {h.min().item():.6f}")
    print(f"  Frac non-zero      = {(h > 0).float().mean().item():.4f}")
    
    # ==== Orbit statistics ====
    # Compute E_g[h(gx)] for each x by sampling many rotations
    orbit_sums = torch.zeros_like(h)
    
    for _ in range(n_orbit_samples):
        theta = sample_rotations(N, device=device)
        x_rot = rotate(data, theta)
        with torch.no_grad():
            h_rot = model.forward_with_intermediate(x_rot, layer_idx)
        orbit_sums += h_rot
    
    orbit_means = orbit_sums / n_orbit_samples  # E_g[h(gx)] for each x, shape (N, D)
    
    # C = E_x[||E_g[h(gx)]||²]
    orbit_mean_sq_norms = (orbit_means ** 2).sum(dim=-1)  # ||E_g[h(gx)]||² for each x
    C = orbit_mean_sq_norms.mean()  # E_x[||E_g[h(gx)]||²]
    
    # Grand orbit mean (should equal h_mean for rotation-invariant data)
    grand_orbit_mean = orbit_means.mean(dim=0)
    
    print(f"\nOrbit statistics:")
    print(f"  E[||E_g h(gx)||²] (C) = {C.item():.6f}")
    print(f"  ||E_x[E_g h(gx)]||²   = {(grand_orbit_mean ** 2).sum().item():.6f}")
    print(f"  (should ≈ B={mean_sq_norm.item():.6f} for rot-invariant data)")
    
    # ==== Compute orbit variance ====
    # E_x[Var_g(h(gx))] via the formula: E_g[||h||²] - ||E_g[h]||²
    orbit_sq_norm_sums = torch.zeros(N, device=device)
    
    for _ in range(n_orbit_samples):
        theta = sample_rotations(N, device=device)
        x_rot = rotate(data, theta)
        with torch.no_grad():
            h_rot = model.forward_with_intermediate(x_rot, layer_idx)
        orbit_sq_norm_sums += (h_rot ** 2).sum(dim=-1)
    
    orbit_sq_norm_means = orbit_sq_norm_sums / n_orbit_samples  # E_g[||h(gx)||²] for each x
    orbit_vars = orbit_sq_norm_means - orbit_mean_sq_norms  # Var_g for each x
    mean_orbit_var = orbit_vars.mean()  # E_x[Var_g(h(gx))]
    
    print(f"\nOrbit variance:")
    print(f"  E_x[Var_g(h(gx))]     = {mean_orbit_var.item():.6f}")
    
    # ==== Compute Q using different methods ====
    
    # Method 1: Pairwise orbit differences (current implementation)
    numerator_pairwise = 0.0
    for _ in range(n_rotations):
        theta1 = sample_rotations(N, device=device)
        theta2 = sample_rotations(N, device=device)
        x_rot1 = rotate(data, theta1)
        x_rot2 = rotate(data, theta2)
        with torch.no_grad():
            h1 = model.forward_with_intermediate(x_rot1, layer_idx)
            h2 = model.forward_with_intermediate(x_rot2, layer_idx)
        numerator_pairwise += ((h1 - h2) ** 2).sum(dim=-1).mean()
    numerator_pairwise /= n_rotations
    Q_pairwise = (numerator_pairwise / 2) / pairwise_var
    
    # Method 2: Direct variance ratio
    Q_direct = mean_orbit_var / standard_var
    
    # Method 3: Using the full formula Q = (A - C) / (A - B)
    A = h_sq_norm_mean
    B = mean_sq_norm
    Q_full_formula = (A - C) / (A - B)
    
    # Method 4: Simplified formula Q = 1 - C/A (only valid if B ≈ 0)
    Q_simplified = 1 - C / A
    
    print(f"\nQ values by method:")
    print(f"  Pairwise (current)    = {Q_pairwise.item():.6f}")
    print(f"  Direct variance ratio = {Q_direct.item():.6f}")
    print(f"  Full: (A-C)/(A-B)     = {Q_full_formula.item():.6f}")
    print(f"  Simplified: 1 - C/A   = {Q_simplified.item():.6f} (only valid if B≈0)")
    
    # ==== Analysis ====
    print(f"\nAnalysis:")
    print(f"  B/A ratio = {(B/A).item():.6f} (how much mean contributes to E[||h||²])")
    print(f"  C/A ratio = {(C/A).item():.6f} (orbit mean contribution)")
    print(f"  C/B ratio = {(C/B).item():.6f} (orbit mean vs data mean squared norms)")
    
    if (B/A).item() > 0.1:
        print(f"\n  WARNING: B/A = {(B/A).item():.3f} is significant!")
        print(f"  The simplified formula Q = 1 - C/A is NOT valid here.")
        print(f"  Difference: simplified - full = {(Q_simplified - Q_full_formula).item():.6f}")
    
    return {
        'A': A.item(),
        'B': B.item(), 
        'C': C.item(),
        'Q_pairwise': Q_pairwise.item(),
        'Q_direct': Q_direct.item(),
        'Q_full_formula': Q_full_formula.item(),
        'Q_simplified': Q_simplified.item(),
    }


def main():
    torch.manual_seed(42)
    
    # Create data: uniform on annulus (rotation-invariant distribution)
    N = 1000
    r_inner, r_outer = 0.5, 2.0
    r = torch.sqrt(torch.rand(N) * (r_outer**2 - r_inner**2) + r_inner**2)
    theta = torch.rand(N) * 2 * np.pi
    data = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    
    print("Data: uniform on annulus, r ∈ [0.5, 2.0]")
    print(f"Data shape: {data.shape}")
    
    # Create model
    dims = [2, 128, 128, 128, 1]
    model = MLP(dims)
    
    print(f"\nModel: MLP with dims {dims}")
    print(f"Random (untrained) weights")
    
    # Test each layer
    for layer_idx in range(1, model.num_linear_layers):
        compute_Q_diagnostic(model, data, layer_idx)
    
    # Also test with centered activations
    print(f"\n{'='*60}")
    print("Testing with CENTERED activations (subtracting mean)")
    print(f"{'='*60}")
    
    class CenteredModel(nn.Module):
        def __init__(self, base_model, layer_idx, train_data):
            super().__init__()
            self.base_model = base_model
            self.layer_idx = layer_idx
            # Compute mean on training data
            with torch.no_grad():
                h = base_model.forward_with_intermediate(train_data, layer_idx)
                self.register_buffer('mean', h.mean(dim=0))
        
        @property
        def num_linear_layers(self):
            return self.base_model.num_linear_layers
        
        def forward_with_intermediate(self, x, layer_idx):
            h = self.base_model.forward_with_intermediate(x, layer_idx)
            if layer_idx == self.layer_idx:
                return h - self.mean
            return h
    
    # Test layer 1 with centering
    centered_model = CenteredModel(model, 1, data)
    compute_Q_diagnostic(centered_model, data, 1)


if __name__ == "__main__":
    main()
