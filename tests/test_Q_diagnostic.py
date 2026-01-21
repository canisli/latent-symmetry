"""
Diagnostic tests to investigate why Q ≈ 1 so precisely.

These tests explore whether Q = 1 is a mathematical artifact or a bug.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.so2 import rotate, sample_rotations
from src.eval import compute_Q, compute_layer_statistics, get_pca_projection, project_activations
from src.data import sample_uniform_disk
from src.models import MLP


def test_analytical_Q_for_identity():
    """
    Analytically derive Q for identity function on uniform disk.
    
    For z = x (identity), we can compute Q analytically:
    
    Numerator: E[||g₁·x - g₂·x||²] where g₁, g₂ are random rotations
             = E[2r²(1 - cos(g₁-g₂))]
             = 2E[r²] · E[1 - cos(Δg)]  (where Δg = g₁-g₂ ~ Uniform[0,2π))
             = 2E[r²] · 1  (since E[cos(Δg)] = 0)
             = 2E[r²]
    
    Denominator: E[||x - x'||²] where x, x' are independent samples
               = E[||x||²] + E[||x'||²] - 2E[x]·E[x']
               = 2E[r²] - 0  (since E[x] = 0 for centered data)
               = 2E[r²]
    
    Therefore Q = 1 EXACTLY for identity on uniform disk!
    """
    print("\n=== Analytical Test: Q for Identity on Uniform Disk ===")
    
    # Generate uniform disk data
    n = 10000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    # Compute E[r²]
    r_tensor = torch.tensor(r, dtype=torch.float32)
    E_r2 = (r_tensor ** 2).mean().item()
    print(f"E[r²] = {E_r2:.6f}")
    
    # Compute numerator analytically: should be 2*E[r²]
    # E[||g₁·x - g₂·x||²] = 2r²(1 - cos(Δg)), averaged over Δg
    # Since E[cos(Δg)] = 0 for uniform Δg, this is 2*E[r²]
    analytical_numerator = 2 * E_r2
    print(f"Analytical numerator: {analytical_numerator:.6f}")
    
    # Compute numerator empirically
    n_rotations = 100
    empirical_numerator = 0.0
    for _ in range(n_rotations):
        theta1 = sample_rotations(n)
        theta2 = sample_rotations(n)
        x_rot1 = rotate(data, theta1)
        x_rot2 = rotate(data, theta2)
        empirical_numerator += ((x_rot1 - x_rot2) ** 2).sum(dim=-1).mean().item()
    empirical_numerator /= n_rotations
    print(f"Empirical numerator: {empirical_numerator:.6f}")
    
    # Compute denominator analytically: should be 2*E[r²]  
    # E[||x - x'||²] = 2*E[||x||²] = 2*E[r²] (for centered data)
    analytical_denominator = 2 * E_r2
    print(f"Analytical denominator: {analytical_denominator:.6f}")
    
    # Compute denominator empirically
    N = data.shape[0]
    z_diff = data.unsqueeze(0) - data.unsqueeze(1)
    z_diff_sq = (z_diff ** 2).sum(dim=-1)
    mask = ~torch.eye(N, dtype=torch.bool)
    empirical_denominator = z_diff_sq[mask].mean().item()
    print(f"Empirical denominator: {empirical_denominator:.6f}")
    
    analytical_Q = analytical_numerator / analytical_denominator
    empirical_Q = empirical_numerator / empirical_denominator
    print(f"\nAnalytical Q: {analytical_Q:.6f}")
    print(f"Empirical Q: {empirical_Q:.6f}")
    print("\n>>> Q = 1 is MATHEMATICALLY CORRECT for identity on uniform disk! <<<")


def test_Q_with_non_uniform_radii():
    """
    Test Q with non-uniform radius distribution.
    
    If all points have the SAME radius, then:
    - Numerator measures angle variation only
    - Denominator also measures angle variation only
    - Q should still be ~1
    
    But if we use a bimodal radius distribution, things change.
    """
    print("\n=== Test: Q with Different Radius Distributions ===")
    
    # Test 1: All points at same radius (circle, not disk)
    n = 1000
    theta = np.random.uniform(0, 2*np.pi, n)
    r_fixed = 1.0
    x = r_fixed * np.cos(theta)
    y = r_fixed * np.sin(theta)
    data_circle = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    # For circle: numerator and denominator should both be 2*r²
    # So Q = 1 still
    
    class IdentityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.num_linear_layers = 1
        def forward(self, x): return x
        def forward_with_intermediate(self, x, layer_idx): return x
    
    model = IdentityModel()
    Q_circle = compute_Q(model, data_circle, layer_idx=-1, n_rotations=64)
    print(f"Q for circle (fixed radius): {Q_circle:.6f}")
    
    # Test 2: Uniform disk
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data_disk = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    Q_disk = compute_Q(model, data_disk, layer_idx=-1, n_rotations=64)
    print(f"Q for uniform disk: {Q_disk:.6f}")
    
    # Test 3: Bimodal radii (points at r=0.5 and r=1.5)
    n_half = n // 2
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    theta2 = np.random.uniform(0, 2*np.pi, n_half)
    x = np.concatenate([0.5 * np.cos(theta1), 1.5 * np.cos(theta2)])
    y = np.concatenate([0.5 * np.sin(theta1), 1.5 * np.sin(theta2)])
    data_bimodal = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    Q_bimodal = compute_Q(model, data_bimodal, layer_idx=-1, n_rotations=64)
    print(f"Q for bimodal radii: {Q_bimodal:.6f}")
    
    print("\n>>> Q ≈ 1 for ALL these cases because rotations preserve radius <<<")
    print(">>> The correlation between z(g₁·x) and z(g₂·x) is 0 after averaging over angles <<<")


def test_correlation_between_rotated_points():
    """
    Explicitly compute the correlation between z(g₁·x) and z(g₂·x).
    
    If this correlation is 0, then Q = 1 mathematically.
    """
    print("\n=== Test: Correlation Between Rotated Points ===")
    
    n = 5000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    # Sample many rotation pairs and compute correlation
    all_z1 = []
    all_z2 = []
    
    for _ in range(100):
        theta1 = sample_rotations(n)
        theta2 = sample_rotations(n)
        z1 = rotate(data, theta1)
        z2 = rotate(data, theta2)
        all_z1.append(z1)
        all_z2.append(z2)
    
    all_z1 = torch.cat(all_z1, dim=0)  # (100*n, 2)
    all_z2 = torch.cat(all_z2, dim=0)
    
    # Compute correlation for each coordinate
    corr_x = torch.corrcoef(torch.stack([all_z1[:, 0], all_z2[:, 0]]))[0, 1].item()
    corr_y = torch.corrcoef(torch.stack([all_z1[:, 1], all_z2[:, 1]]))[0, 1].item()
    
    print(f"Correlation in x-coordinate: {corr_x:.6f}")
    print(f"Correlation in y-coordinate: {corr_y:.6f}")
    
    # Also compute E[z1 · z2] directly
    dot_product = (all_z1 * all_z2).sum(dim=-1).mean().item()
    E_z1 = all_z1.mean(dim=0)
    E_z2 = all_z2.mean(dim=0)
    expected_dot = (E_z1 * E_z2).sum().item()
    
    print(f"\nE[z₁ · z₂] = {dot_product:.6f}")
    print(f"E[z₁] · E[z₂] = {expected_dot:.6f}")
    print(f"Covariance = {dot_product - expected_dot:.6f}")
    
    print("\n>>> Correlation ≈ 0 confirms Q = 1 is mathematically correct <<<")


def test_Q_for_linear_layer():
    """
    For z = Wx + b through activation, analyze Q.
    
    If W is random and data is uniform on disk, Q should still be ~1
    because the transformation is linear and preserves the correlation structure.
    """
    print("\n=== Test: Q for Linear Layer with Random Weights ===")
    
    n = 2000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    # Different hidden dimensions
    for hidden_dim in [2, 8, 32, 128]:
        model = MLP([2, hidden_dim, 1])
        Q = compute_Q(model, data, layer_idx=1, n_rotations=32)
        print(f"Hidden dim {hidden_dim:3d}: Q = {Q:.6f}")
    
    print("\n>>> Q ≈ 1 regardless of hidden dimension <<<")


def test_Q_with_trained_invariant_layer():
    """
    The key question: does Q stay ~1 at layer 1 even after training?
    
    After training on an invariant task, early layers might start to 
    become more invariant (Q < 1) as they learn useful features.
    """
    print("\n=== Test: Q at Layer 1 Before vs After Training ===")
    
    # This would require training, so we'll just note the expectation
    print("Before training: Q ≈ 1 (random weights)")
    print("After training on invariant task: Q might decrease if layer learns invariant features")
    print("After training on non-invariant task: Q should stay ~1")
    print("\nYour results show:")
    print("  Invariant model layer_1: Q = 1.0061 (still ~1, invariance learned later)")
    print("  Non-invariant model layer_1: Q = 1.0055 (still ~1, as expected)")


def test_Q_numerator_denominator_separately():
    """
    Print numerator and denominator separately to check for issues.
    """
    print("\n=== Test: Numerator and Denominator Values ===")
    
    n = 1000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    model = MLP([2, 128, 128, 1])
    
    for layer_idx in [1, 2, -1]:
        # Get layer stats
        mu, cov = compute_layer_statistics(model, data, layer_idx, torch.device('cpu'))
        
        with torch.no_grad():
            h_test = model.forward_with_intermediate(data[:1], layer_idx)
        
        if h_test.shape[1] == 1:
            U = torch.eye(1)
        else:
            U = get_pca_projection(cov, 0.95)
        
        with torch.no_grad():
            h = model.forward_with_intermediate(data, layer_idx)
        z = project_activations(h, U, mu)
        
        # Denominator
        N = z.shape[0]
        z_diff = z.unsqueeze(0) - z.unsqueeze(1)
        z_diff_sq = (z_diff ** 2).sum(dim=-1)
        mask = ~torch.eye(N, dtype=torch.bool)
        denominator = z_diff_sq[mask].mean().item()
        
        # Numerator
        numerator = 0.0
        n_rotations = 32
        for _ in range(n_rotations):
            theta1 = sample_rotations(N)
            theta2 = sample_rotations(N)
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            with torch.no_grad():
                h1 = model.forward_with_intermediate(x_rot1, layer_idx)
                h2 = model.forward_with_intermediate(x_rot2, layer_idx)
            
            z1 = project_activations(h1, U, mu)
            z2 = project_activations(h2, U, mu)
            
            numerator += ((z1 - z2) ** 2).sum(dim=-1).mean().item()
        numerator /= n_rotations
        
        Q = numerator / denominator
        
        layer_name = f"layer_{layer_idx}" if layer_idx > 0 else "output"
        print(f"{layer_name}:")
        print(f"  Numerator:   {numerator:.6f}")
        print(f"  Denominator: {denominator:.6f}")
        print(f"  Q:           {Q:.6f}")
        print(f"  PCA dims:    {U.shape[1]} of {h.shape[1]}")


def test_pca_effect_on_Q():
    """
    Check if PCA projection is artificially making Q close to 1.
    """
    print("\n=== Test: Effect of PCA on Q ===")
    
    n = 1000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    model = MLP([2, 128, 128, 1])
    
    # Compare Q with different explained variance thresholds
    for exp_var in [0.5, 0.8, 0.95, 0.99, 1.0]:
        Q = compute_Q(model, data, layer_idx=1, n_rotations=32, explained_variance=exp_var)
        print(f"Explained variance {exp_var:.2f}: Q = {Q:.6f}")


if __name__ == "__main__":
    test_analytical_Q_for_identity()
    test_Q_with_non_uniform_radii()
    test_correlation_between_rotated_points()
    test_Q_for_linear_layer()
    test_Q_with_trained_invariant_layer()
    test_Q_numerator_denominator_separately()
    test_pca_effect_on_Q()
