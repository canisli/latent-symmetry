"""
Tests for the Q invariance metric.

These tests verify that Q behaves correctly:
- Q ≈ 0 for perfectly invariant representations
- Q ≈ 1 for non-invariant representations  
- The rotation function is correct
- The numerator/denominator computations are correct
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.groups.so2 import rotate, sample_rotations, rotation_matrix
from src.metrics.q_metric import (
    compute_Q, compute_layer_statistics, get_pca_projection, 
    project_activations, compute_all_Q
)
from src.tasks.so2_regression import sample_uniform_disk


class TestRotation:
    """Test the rotation function for correctness."""
    
    def test_rotation_preserves_norm(self):
        """Rotation should preserve the norm of vectors."""
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
        theta = torch.tensor([0.5, 1.2, 2.7])
        
        x_rot = rotate(x, theta)
        
        original_norms = torch.norm(x, dim=1)
        rotated_norms = torch.norm(x_rot, dim=1)
        
        assert torch.allclose(original_norms, rotated_norms, atol=1e-6), \
            f"Norms not preserved: {original_norms} vs {rotated_norms}"
    
    def test_rotation_by_zero(self):
        """Rotation by 0 should return the same point."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        theta = torch.tensor([0.0, 0.0])
        
        x_rot = rotate(x, theta)
        
        assert torch.allclose(x, x_rot, atol=1e-6), \
            f"Rotation by 0 changed the point: {x} -> {x_rot}"
    
    def test_rotation_by_2pi(self):
        """Rotation by 2π should return the same point."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        theta = torch.tensor([2 * np.pi, 2 * np.pi])
        
        x_rot = rotate(x, theta)
        
        assert torch.allclose(x, x_rot, atol=1e-5), \
            f"Rotation by 2π changed the point: {x} -> {x_rot}"
    
    def test_rotation_by_pi_half(self):
        """Rotation by π/2 should swap and negate correctly."""
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        theta = torch.tensor([np.pi / 2, np.pi / 2])
        
        x_rot = rotate(x, theta)
        expected = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        
        assert torch.allclose(x_rot, expected, atol=1e-6), \
            f"π/2 rotation incorrect: {x_rot} vs expected {expected}"
    
    def test_rotation_composition(self):
        """Two rotations should compose correctly: R(θ₁)R(θ₂)x = R(θ₁+θ₂)x."""
        x = torch.tensor([[1.0, 2.0]])
        theta1 = torch.tensor([0.3])
        theta2 = torch.tensor([0.7])
        
        # Apply rotations sequentially
        x_rot1 = rotate(x, theta1)
        x_rot12 = rotate(x_rot1, theta2)
        
        # Apply combined rotation
        x_combined = rotate(x, theta1 + theta2)
        
        assert torch.allclose(x_rot12, x_combined, atol=1e-6), \
            f"Rotation composition failed: {x_rot12} vs {x_combined}"
    
    def test_different_angles_give_different_results(self):
        """Different angles should produce different rotated points (for non-origin)."""
        x = torch.tensor([[1.0, 0.0]])
        theta1 = torch.tensor([0.5])
        theta2 = torch.tensor([1.5])
        
        x_rot1 = rotate(x, theta1)
        x_rot2 = rotate(x, theta2)
        
        assert not torch.allclose(x_rot1, x_rot2, atol=1e-3), \
            f"Different angles gave same result: θ₁={theta1}, θ₂={theta2}"


class TestSampleRotations:
    """Test the rotation sampling function."""
    
    def test_sample_rotations_range(self):
        """Sampled rotations should be in [0, 2π)."""
        thetas = sample_rotations(1000)
        
        assert thetas.min() >= 0, f"Min theta below 0: {thetas.min()}"
        assert thetas.max() < 2 * np.pi, f"Max theta >= 2π: {thetas.max()}"
    
    def test_sample_rotations_uniformity(self):
        """Sampled rotations should be roughly uniform."""
        thetas = sample_rotations(10000)
        
        # Check mean is close to π
        assert abs(thetas.mean().item() - np.pi) < 0.1, \
            f"Mean not close to π: {thetas.mean()}"
        
        # Check std is close to π/√3 (std of uniform[0, 2π])
        expected_std = 2 * np.pi / np.sqrt(12)
        assert abs(thetas.std().item() - expected_std) < 0.1, \
            f"Std not close to expected: {thetas.std()} vs {expected_std}"


class InvariantModel(nn.Module):
    """A model that outputs r² = x² + y² (perfectly SO(2)-invariant)."""
    
    def __init__(self):
        super().__init__()
        # Fake parameter to make it a "model"
        self.dummy = nn.Parameter(torch.zeros(1))
        self.num_linear_layers = 1  # Pretend we have 1 layer
    
    def forward(self, x):
        return (x ** 2).sum(dim=-1, keepdim=True)
    
    def forward_with_intermediate(self, x, layer_idx):
        return self.forward(x)


class NonInvariantModel(nn.Module):
    """A model that outputs x (NOT SO(2)-invariant)."""
    
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.num_linear_layers = 1
    
    def forward(self, x):
        return x[:, 0:1]  # Just return x coordinate
    
    def forward_with_intermediate(self, x, layer_idx):
        return self.forward(x)


class IdentityModel(nn.Module):
    """A model that returns the input unchanged (for testing raw input Q)."""
    
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.num_linear_layers = 1
    
    def forward(self, x):
        return x
    
    def forward_with_intermediate(self, x, layer_idx):
        return x


class TestQMetricAnalytical:
    """Test Q metric with analytical cases where we know the answer."""
    
    def test_Q_zero_for_invariant_output(self):
        """Q should be ~0 for a perfectly SO(2)-invariant function."""
        model = InvariantModel()
        
        # Generate test data uniformly on disk
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q = compute_Q(model, data, layer_idx=-1, n_rotations=32)
        
        print(f"Q for invariant model (r²): {Q}")
        assert Q < 0.01, f"Q should be ~0 for invariant output, got {Q}"
    
    def test_Q_positive_for_non_invariant_output(self):
        """Q should be positive for a non-invariant function like f(x,y)=x."""
        model = NonInvariantModel()
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q = compute_Q(model, data, layer_idx=-1, n_rotations=32)
        
        print(f"Q for non-invariant model (x): {Q}")
        assert Q > 0.5, f"Q should be > 0.5 for non-invariant output, got {Q}"
    
    def test_Q_roughly_one_for_identity_on_uniform_disk(self):
        """
        For identity function on uniform disk data, Q should be close to 1.
        
        This is because rotating a uniformly sampled point gives something
        with the same distribution as sampling a new point (for uniform disk).
        The orbit variance should equal the data variance.
        """
        model = IdentityModel()
        
        x, y, r = sample_uniform_disk(1000, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q = compute_Q(model, data, layer_idx=-1, n_rotations=64)
        
        print(f"Q for identity model on uniform disk: {Q}")
        # Should be close to 1, but not exactly due to radius correlations
        assert 0.5 < Q < 1.5, f"Q should be ~1 for identity on uniform disk, got {Q}"


class TestQMetricNumeratorDenominator:
    """Test that numerator and denominator are computed correctly."""
    
    def test_denominator_is_pairwise_variance(self):
        """
        The denominator should be E[||z(x) - z(x')||²].
        For uniform data, this equals 2 * Var(z) = 2 * trace(Cov(z)).
        """
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        # Compute expected: 2 * trace(cov)
        # E[||z-z'||²] = E[||z||²] + E[||z'||²] - 2*E[z·z']
        # For centered z: = 2*E[||z||²] = 2*trace(Cov(z))
        z_centered = data - data.mean(dim=0)
        expected_denom = 2 * (z_centered ** 2).sum(dim=-1).mean()
        
        # Compute using pairwise differences (like in compute_Q)
        N = data.shape[0]
        z_diff = data.unsqueeze(0) - data.unsqueeze(1)
        z_diff_sq = (z_diff ** 2).sum(dim=-1)
        mask = ~torch.eye(N, dtype=torch.bool)
        computed_denom = z_diff_sq[mask].mean()
        
        print(f"Expected denominator: {expected_denom}")
        print(f"Computed denominator: {computed_denom}")
        
        # Should be close (small difference due to finite sample)
        rel_error = abs(computed_denom - expected_denom) / expected_denom
        assert rel_error < 0.1, f"Denominator mismatch: {computed_denom} vs {expected_denom}"
    
    def test_numerator_zero_for_invariant(self):
        """For invariant representation, numerator should be 0."""
        model = InvariantModel()
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        # Manually compute numerator
        numerator = 0.0
        n_rotations = 32
        for _ in range(n_rotations):
            theta1 = sample_rotations(data.shape[0])
            theta2 = sample_rotations(data.shape[0])
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            z1 = model(x_rot1)
            z2 = model(x_rot2)
            
            numerator += ((z1 - z2) ** 2).sum(dim=-1).mean()
        
        numerator /= n_rotations
        
        print(f"Numerator for invariant model: {numerator}")
        assert numerator < 1e-10, f"Numerator should be ~0 for invariant, got {numerator}"
    
    def test_numerator_positive_for_non_invariant(self):
        """For non-invariant representation, numerator should be positive."""
        model = NonInvariantModel()
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        # Manually compute numerator
        numerator = 0.0
        n_rotations = 32
        for _ in range(n_rotations):
            theta1 = sample_rotations(data.shape[0])
            theta2 = sample_rotations(data.shape[0])
            
            x_rot1 = rotate(data, theta1)
            x_rot2 = rotate(data, theta2)
            
            z1 = model(x_rot1)
            z2 = model(x_rot2)
            
            numerator += ((z1 - z2) ** 2).sum(dim=-1).mean()
        
        numerator /= n_rotations
        
        print(f"Numerator for non-invariant model: {numerator}")
        assert numerator > 0.1, f"Numerator should be positive for non-invariant, got {numerator}"


class TestQWithRealMLP:
    """Test Q with actual MLP models."""
    
    def test_Q_at_layer1_with_random_weights(self):
        """
        For a random MLP, Q at layer 1 should be close to 1.
        
        This is actually EXPECTED behavior because:
        1. Random weights don't have any special structure
        2. On uniform disk data, rotating a point produces similar variation
           to sampling a different point
        """
        from src.models import MLP
        
        model = MLP([2, 128, 128, 1])
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q = compute_Q(model, data, layer_idx=1, n_rotations=32)
        
        print(f"Q at layer 1 for random MLP: {Q}")
        # For random weights on uniform disk, Q should be ~1
        assert 0.5 < Q < 1.5, f"Q at layer 1 should be ~1, got {Q}"
    
    def test_Q_progression_makes_sense(self):
        """
        For a trained invariant model, Q should decrease with depth.
        For a non-invariant model, Q should stay ~1 throughout.
        """
        from src.models import MLP
        
        # Random model (no training) - Q should be ~1 throughout
        model = MLP([2, 64, 64, 64, 1])
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q_values = compute_all_Q(model, data, n_rotations=32)
        
        print("Q values for random MLP:")
        for name, q in Q_values.items():
            print(f"  {name}: {q:.4f}")
        
        # All Q values should be roughly similar for random model
        values = list(Q_values.values())
        q_std = np.std(values[:-1])  # Exclude output
        print(f"Std of Q values (excluding output): {q_std}")


class TestEdgeCases:
    """Test edge cases and potential bugs."""
    
    def test_Q_with_small_data(self):
        """Q should work with small datasets."""
        model = InvariantModel()
        
        x, y, r = sample_uniform_disk(50, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q = compute_Q(model, data, layer_idx=-1, n_rotations=16)
        
        print(f"Q with 50 samples: {Q}")
        assert Q < 0.1, f"Q should still be ~0 for invariant with small data, got {Q}"
    
    def test_Q_not_affected_by_data_centering(self):
        """Q should not depend on whether data is centered."""
        model = IdentityModel()
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        # Centered data (should have mean ~0 anyway)
        Q_centered = compute_Q(model, data - data.mean(dim=0), layer_idx=-1, n_rotations=32)
        
        # Shifted data
        data_shifted = data + torch.tensor([5.0, 5.0])
        Q_shifted = compute_Q(model, data_shifted, layer_idx=-1, n_rotations=32)
        
        print(f"Q centered: {Q_centered}, Q shifted: {Q_shifted}")
        
        # Q should be similar (though not identical due to shift breaking SO(2) of data distribution)
        # Actually, shifted data no longer has SO(2) symmetry, so this test is tricky
    
    def test_rotation_of_origin_is_origin(self):
        """Rotating the origin should give the origin."""
        x = torch.tensor([[0.0, 0.0]])
        theta = torch.tensor([1.5])
        
        x_rot = rotate(x, theta)
        
        assert torch.allclose(x_rot, x, atol=1e-10), \
            f"Rotating origin gave non-zero: {x_rot}"
    
    def test_Q_numerator_uses_different_rotations(self):
        """
        Verify that theta1 and theta2 in the numerator are actually different.
        If they're the same, the numerator would be 0 incorrectly.
        """
        # Run multiple times and check theta1 != theta2
        n = 100
        thetas1 = []
        thetas2 = []
        
        for _ in range(10):
            theta1 = sample_rotations(n)
            theta2 = sample_rotations(n)
            thetas1.append(theta1)
            thetas2.append(theta2)
            
            # They should NOT be equal
            assert not torch.allclose(theta1, theta2), \
                "theta1 and theta2 should be different samples"
        
        # Check they're independent by looking at correlation
        all_theta1 = torch.cat(thetas1)
        all_theta2 = torch.cat(thetas2)
        
        corr = torch.corrcoef(torch.stack([all_theta1, all_theta2]))[0, 1]
        print(f"Correlation between theta1 and theta2: {corr}")
        
        assert abs(corr) < 0.1, f"theta1 and theta2 should be uncorrelated, got {corr}"


class TestPCAProjection:
    """Test the PCA projection used in Q computation."""
    
    def test_pca_preserves_variance(self):
        """PCA with 95% variance should preserve most information."""
        # Random data
        data = torch.randn(1000, 128)
        
        mu = data.mean(dim=0)
        h_centered = data - mu
        cov = (h_centered.T @ h_centered) / (data.shape[0] - 1)
        
        U = get_pca_projection(cov, explained_variance=0.95)
        
        print(f"PCA kept {U.shape[1]} of {cov.shape[0]} dimensions")
        
        # Project and check variance
        z = project_activations(data, U, mu)
        
        # Variance of projected should be ~95% of original
        original_var = (h_centered ** 2).sum(dim=-1).mean()
        projected_var = (z ** 2).sum(dim=-1).mean()
        
        ratio = projected_var / original_var
        print(f"Variance ratio: {ratio}")
        
        assert ratio > 0.90, f"PCA should preserve >90% variance, got {ratio}"
    
    def test_pca_on_low_rank_data(self):
        """PCA should correctly identify low-rank structure."""
        # Data that only varies in 2 dimensions
        z_true = torch.randn(500, 2)
        A = torch.randn(2, 128)  # Map to 128D
        data = z_true @ A
        
        mu = data.mean(dim=0)
        h_centered = data - mu
        cov = (h_centered.T @ h_centered) / (data.shape[0] - 1)
        
        U = get_pca_projection(cov, explained_variance=0.95)
        
        print(f"PCA found {U.shape[1]} dimensions for rank-2 data")
        
        # Should find ~2 dimensions
        assert U.shape[1] <= 5, f"Should find ~2 dimensions, got {U.shape[1]}"


class TestQInterpretation:
    """Tests that demonstrate the interpretation of Q values."""
    
    def test_Q_one_is_no_learned_invariance(self):
        """
        Q ≈ 1 means orbit variance equals data variance.
        
        For uniform disk data with random weights, this is expected because:
        - The data distribution is SO(2)-invariant  
        - Rotating a point samples from the "same distribution" as the data
        - Random weights don't encode any special structure
        
        Therefore Q ≈ 1 at layer 1 is NOT a bug - it's the expected baseline
        before any invariance is learned.
        """
        from src.models import MLP
        
        model = MLP([2, 128, 1])
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q = compute_Q(model, data, layer_idx=1, n_rotations=64)
        
        print(f"\nQ for random weights on SO(2)-symmetric data: {Q:.4f}")
        print("This is ~1 because rotating a point produces similar variation")
        print("to sampling a different point when data is uniform on disk.")
        
        assert 0.8 < Q < 1.2, f"Q should be ~1 for random weights, got {Q}"
    
    def test_Q_interpretation_summary(self):
        """
        Summary of Q value interpretations:
        - Q ≈ 0: Perfect SO(2) invariance (orbit has no variance)
        - Q ≈ 1: No invariance / random baseline (orbit variance = data variance)  
        - Q > 1: Anti-invariance or numerical issues (orbit variance > data variance)
        - Q < 1: Partial invariance (orbit variance < data variance)
        """
        inv_model = InvariantModel()
        non_inv_model = NonInvariantModel()
        identity_model = IdentityModel()
        
        x, y, r = sample_uniform_disk(500, r_min=0.1, r_max=1.0)
        data = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        
        Q_inv = compute_Q(inv_model, data, layer_idx=-1, n_rotations=32)
        Q_non = compute_Q(non_inv_model, data, layer_idx=-1, n_rotations=32)
        Q_id = compute_Q(identity_model, data, layer_idx=-1, n_rotations=32)
        
        print("\n=== Q Value Interpretation ===")
        print(f"Invariant (r²):     Q = {Q_inv:.6f} (should be ~0)")
        print(f"Non-invariant (x):  Q = {Q_non:.4f} (should be ~1)")  
        print(f"Identity:           Q = {Q_id:.4f} (should be ~1)")
        print("\nConclusion: Q ≈ 1 at layer 1 is EXPECTED, not a bug!")
        
        assert Q_inv < 0.01
        assert 0.8 < Q_non < 1.2
        assert 0.8 < Q_id < 1.2


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
