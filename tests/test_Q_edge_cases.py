"""
Additional edge case tests to verify Q metric behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.groups.so2 import rotate, sample_rotations
from src.metrics.q_metric import compute_Q, compute_all_Q
from src.tasks.so2_regression import sample_uniform_disk
from src.models import MLP


class RadiusModel(nn.Module):
    """Model that computes r = sqrt(x² + y²) at every layer."""
    
    def __init__(self, n_layers=4, hidden_dim=128):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.num_linear_layers = n_layers
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        r = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True))
        return r
    
    def forward_with_intermediate(self, x, layer_idx):
        # All intermediate representations are just r repeated
        r = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True))
        if layer_idx == -1:
            return r
        # Return r repeated to simulate hidden dim
        return r.repeat(1, self.hidden_dim)


class AngleModel(nn.Module):
    """Model that computes angle θ = atan2(y, x) - NOT invariant."""
    
    def __init__(self, n_layers=4, hidden_dim=128):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.num_linear_layers = n_layers
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        theta = torch.atan2(x[:, 1:2], x[:, 0:1])
        return theta
    
    def forward_with_intermediate(self, x, layer_idx):
        theta = torch.atan2(x[:, 1:2], x[:, 0:1])
        if layer_idx == -1:
            return theta
        return theta.repeat(1, self.hidden_dim)


class MixedModel(nn.Module):
    """
    Model that computes mix of r and θ depending on layer:
    - Early layers: more θ (angle) - should have Q ≈ 1
    - Later layers: more r (radius) - should have Q → 0
    """
    
    def __init__(self, n_layers=8, hidden_dim=64):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.num_linear_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    def forward(self, x):
        r = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True))
        return r
    
    def forward_with_intermediate(self, x, layer_idx):
        if layer_idx == -1:
            return self.forward(x)
        
        r = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True))
        theta = torch.atan2(x[:, 1:2], x[:, 0:1])
        
        # Gradually transition from θ to r
        alpha = layer_idx / self.n_layers  # 0 to 1
        
        # Mix: (1-alpha)*[theta features] + alpha*[r features]
        half = self.hidden_dim // 2
        
        # theta part: encode angle in multiple ways (all non-invariant)
        theta_features = torch.cat([
            torch.sin(theta * k) for k in range(1, half + 1)
        ], dim=-1)
        
        # r part: encode radius in multiple ways (all invariant)
        r_features = torch.cat([
            r ** k for k in range(1, half + 1)
        ], dim=-1)
        
        # Blend based on layer depth
        features = (1 - alpha) * theta_features + alpha * r_features
        
        return features


def test_Q_for_radius_model():
    """Q should be 0 at all layers for a model computing only radius."""
    print("\n=== Test: Q for Radius-Only Model ===")
    
    n = 1000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    model = RadiusModel(n_layers=4, hidden_dim=128)
    Q_values = compute_all_Q(model, data, n_rotations=32)
    
    print("Q values (should all be ~0):")
    for name, q in Q_values.items():
        print(f"  {name}: Q = {q:.6f}")
    
    # All should be ~0
    for name, q in Q_values.items():
        assert q < 0.01, f"{name} should have Q ≈ 0, got {q}"
    
    print("✓ All Q values are ~0 as expected")


def test_Q_for_angle_model():
    """Q should be ~1 at all layers for a model computing only angle."""
    print("\n=== Test: Q for Angle-Only Model ===")
    
    n = 1000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    model = AngleModel(n_layers=4, hidden_dim=128)
    Q_values = compute_all_Q(model, data, n_rotations=32)
    
    print("Q values (should all be ~1):")
    for name, q in Q_values.items():
        print(f"  {name}: Q = {q:.6f}")
    
    # All should be ~1
    for name, q in Q_values.items():
        assert 0.5 < q < 1.5, f"{name} should have Q ≈ 1, got {q}"
    
    print("✓ All Q values are ~1 as expected")


def test_Q_for_mixed_model():
    """Q should decrease from ~1 to ~0 as we go deeper in the mixed model."""
    print("\n=== Test: Q for Mixed Model (θ → r transition) ===")
    
    n = 1000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    model = MixedModel(n_layers=8, hidden_dim=64)
    Q_values = compute_all_Q(model, data, n_rotations=32)
    
    print("Q values (should decrease from ~1 to ~0):")
    for name, q in Q_values.items():
        print(f"  {name}: Q = {q:.6f}")
    
    # Check that Q decreases
    values = list(Q_values.values())
    first_half_mean = np.mean(values[:4])
    second_half_mean = np.mean(values[4:])
    
    print(f"\nFirst half mean Q: {first_half_mean:.4f}")
    print(f"Second half mean Q: {second_half_mean:.4f}")
    
    assert first_half_mean > second_half_mean, \
        f"Q should decrease, but first half ({first_half_mean}) <= second half ({second_half_mean})"
    
    print("✓ Q decreases with depth as expected")


def test_Q_without_pca():
    """
    Test Q computation without PCA (explained_variance=1.0).
    
    This checks if PCA is artificially affecting Q.
    """
    print("\n=== Test: Q With vs Without PCA ===")
    
    n = 1000
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    model = MLP([2, 64, 64, 64, 1])
    
    print("Layer 1:")
    for exp_var in [0.5, 0.95, 1.0]:
        Q = compute_Q(model, data, layer_idx=1, n_rotations=32, explained_variance=exp_var)
        print(f"  explained_variance={exp_var:.2f}: Q = {Q:.6f}")
    
    print("\nLayer 2:")
    for exp_var in [0.5, 0.95, 1.0]:
        Q = compute_Q(model, data, layer_idx=2, n_rotations=32, explained_variance=exp_var)
        print(f"  explained_variance={exp_var:.2f}: Q = {Q:.6f}")


def test_Q_stable_across_runs():
    """Test that Q values are stable across multiple runs."""
    print("\n=== Test: Q Stability Across Runs ===")
    
    n = 500
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0, rng=np.random.default_rng(42))
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    torch.manual_seed(123)
    model = MLP([2, 64, 64, 1])
    
    Q_runs = []
    for i in range(5):
        Q = compute_Q(model, data, layer_idx=1, n_rotations=32)
        Q_runs.append(Q)
        print(f"  Run {i+1}: Q = {Q:.6f}")
    
    std = np.std(Q_runs)
    print(f"\nStd across runs: {std:.6f}")
    assert std < 0.05, f"Q should be stable, but std = {std}"
    print("✓ Q is stable across runs")


def test_verify_rotation_changes_activations():
    """
    Verify that rotation actually changes the activations.
    If activations don't change, there might be a bug.
    """
    print("\n=== Test: Verify Rotation Changes Activations ===")
    
    n = 100
    x, y, r = sample_uniform_disk(n, r_min=0.1, r_max=1.0)
    data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    
    model = MLP([2, 64, 64, 1])
    model.eval()
    
    # Get activations for original data
    with torch.no_grad():
        h_orig = model.forward_with_intermediate(data, layer_idx=1)
    
    # Rotate by a fixed angle and get activations
    theta = torch.full((n,), np.pi / 4)  # 45 degrees
    data_rot = rotate(data, theta)
    
    with torch.no_grad():
        h_rot = model.forward_with_intermediate(data_rot, layer_idx=1)
    
    # Check that activations changed
    diff = (h_orig - h_rot).abs().mean().item()
    print(f"Mean absolute difference after 45° rotation: {diff:.6f}")
    
    assert diff > 0.01, f"Activations should change after rotation, but diff = {diff}"
    print("✓ Activations change after rotation")
    
    # Also check output
    with torch.no_grad():
        out_orig = model(data)
        out_rot = model(data_rot)
    
    out_diff = (out_orig - out_rot).abs().mean().item()
    print(f"Mean output difference after 45° rotation: {out_diff:.6f}")


if __name__ == "__main__":
    test_Q_for_radius_model()
    test_Q_for_angle_model()
    test_Q_for_mixed_model()
    test_Q_without_pca()
    test_Q_stable_across_runs()
    test_verify_rotation_changes_activations()
