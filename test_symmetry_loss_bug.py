#!/usr/bin/env python3
"""
Test the symmetry loss computation for potential bugs.

This script checks:
1. Are Lorentz transformations L1 and L2 actually different?
2. Is the symmetry loss correctly detecting non-invariance?
3. Is a truly invariant function (like mass squared) giving zero loss?
4. Is there something wrong with how the loss is computed?
"""

import torch
import numpy as np
from symmetry import rand_lorentz, lorentz_orbit_variance_loss
from models import DeepSets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_lorentz_transforms_are_different():
    """Test that L1 and L2 are actually different transformations."""
    print("\n" + "="*60)
    print("TEST 1: Are L1 and L2 different?")
    print("="*60)
    
    B = 4
    generator = torch.Generator(device=device).manual_seed(42)
    
    L1 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=0.5,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    L2 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=0.5,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    
    diff = (L1 - L2).abs().max().item()
    print(f"Max absolute difference between L1 and L2: {diff:.6f}")
    
    if diff < 1e-6:
        print("BUG FOUND: L1 and L2 are identical!")
        return False
    else:
        print("OK: L1 and L2 are different")
        return True


def test_lorentz_preserves_mass():
    """Test that Lorentz transformations preserve the mass squared."""
    print("\n" + "="*60)
    print("TEST 2: Do Lorentz transforms preserve mass?")
    print("="*60)
    
    B, N = 4, 10
    
    # Create some random 4-vectors with non-zero mass
    E = torch.rand(B, N, 1) * 10 + 5  # Energy between 5 and 15
    p = torch.randn(B, N, 3) * 2  # Spatial momentum
    x = torch.cat([E, p], dim=-1).to(device)
    
    # Compute original mass squared: m² = E² - |p|²
    m2_orig = x[..., 0]**2 - (x[..., 1:]**2).sum(dim=-1)
    
    # Apply Lorentz transformation
    L = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=0.5,
        device=device,
        dtype=torch.float32,
    )
    x_rot = torch.matmul(L.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    # Compute transformed mass squared
    m2_rot = x_rot[..., 0]**2 - (x_rot[..., 1:]**2).sum(dim=-1)
    
    diff = (m2_orig - m2_rot).abs().max().item()
    rel_diff = (diff / m2_orig.abs().max().item()) * 100
    
    print(f"Max absolute difference in m²: {diff:.6e}")
    print(f"Relative difference: {rel_diff:.4f}%")
    
    if rel_diff > 1.0:
        print("WARNING: Lorentz transformation is not preserving mass correctly!")
        return False
    else:
        print("OK: Mass is preserved under Lorentz transformation")
        return True


def test_invariant_function_gives_zero_loss():
    """Test that a truly invariant function gives zero symmetry loss."""
    print("\n" + "="*60)
    print("TEST 3: Does mass² (invariant) give zero loss?")
    print("="*60)
    
    class MassSquaredModel(torch.nn.Module):
        """A model that computes mass squared - should be Lorentz invariant."""
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)  # Need some parameters
        
        def forward(self, x):
            # x: (B, N, 4) -> m²: (B, N, 1)
            m2 = x[..., 0:1]**2 - (x[..., 1:]**2).sum(dim=-1, keepdim=True)
            return m2.sum(dim=1)  # Sum over particles: (B, 1)
        
        def forward_with_intermediate(self, x, layer_idx, mask=None):
            # Return per-particle m² for any layer_idx
            m2 = x[..., 0:1]**2 - (x[..., 1:]**2).sum(dim=-1, keepdim=True)
            return m2  # (B, N, 1)
    
    model = MassSquaredModel().to(device)
    
    B, N = 32, 128
    x = torch.randn(B, N, 4).to(device)
    x[..., 0] = x[..., 0].abs() + 1  # Make energy positive
    
    loss = lorentz_orbit_variance_loss(model, x, layer_idx=1, std_eta=0.5)
    print(f"Symmetry loss for m² (should be ~0): {loss.item():.6e}")
    
    if loss.item() > 1e-4:
        print("WARNING: Invariant function gives non-zero loss!")
        return False
    else:
        print("OK: Invariant function gives near-zero loss")
        return True


def test_non_invariant_function_gives_nonzero_loss():
    """Test that a non-invariant function gives non-zero symmetry loss."""
    print("\n" + "="*60)
    print("TEST 4: Does energy (non-invariant) give non-zero loss?")
    print("="*60)
    
    class EnergyModel(torch.nn.Module):
        """A model that returns energy - NOT Lorentz invariant."""
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)
        
        def forward(self, x):
            return x[..., 0:1].sum(dim=1)  # Sum of energies
        
        def forward_with_intermediate(self, x, layer_idx, mask=None):
            return x[..., 0:1]  # Just energy: (B, N, 1)
    
    model = EnergyModel().to(device)
    
    B, N = 32, 128
    x = torch.randn(B, N, 4).to(device)
    x[..., 0] = x[..., 0].abs() + 1
    
    loss = lorentz_orbit_variance_loss(model, x, layer_idx=1, std_eta=0.5)
    print(f"Symmetry loss for energy (should be >0): {loss.item():.6e}")
    
    if loss.item() < 1e-4:
        print("BUG FOUND: Non-invariant function gives near-zero loss!")
        return False
    else:
        print("OK: Non-invariant function gives non-zero loss")
        return True


def test_deepsets_layer1_non_invariance():
    """Test that a random DeepSets model has non-zero layer 1 symmetry loss."""
    print("\n" + "="*60)
    print("TEST 5: Does random DeepSets have non-zero layer 1 loss?")
    print("="*60)
    
    model = DeepSets(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=4,
        num_rho_layers=4,
        pool_mode='sum',
    ).to(device)
    
    B, N = 32, 128
    x = torch.randn(B, N, 4).to(device)
    x[..., 0] = x[..., 0].abs() + 1
    
    loss = lorentz_orbit_variance_loss(model, x, layer_idx=1, std_eta=0.5)
    print(f"Symmetry loss for random DeepSets layer 1: {loss.item():.6e}")
    
    if loss.item() < 1e-2:
        print("WARNING: Random model has suspiciously low symmetry loss!")
        return False
    else:
        print("OK: Random model has non-trivial symmetry loss")
        return True


def test_mask_computation():
    """Test that the mask is being computed correctly."""
    print("\n" + "="*60)
    print("TEST 6: Is mask computed correctly?")
    print("="*60)
    
    model = DeepSets(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=4,
        num_rho_layers=4,
        pool_mode='sum',
    ).to(device)
    
    B, N = 4, 10
    x = torch.randn(B, N, 4).to(device)
    
    # Zero out some particles (padding)
    x[:, 5:, :] = 0.0
    
    # The mask should have 5 True values per batch
    mask = torch.any(x != 0.0, dim=-1)
    print(f"Mask sum per batch (should be 5): {mask.sum(dim=1).tolist()}")
    
    # Apply Lorentz transformation
    L = rand_lorentz(shape=torch.Size([B]), std_eta=0.5, device=device, dtype=torch.float32)
    x_rot = torch.matmul(L.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    # Check if zero particles remain zero after transformation
    # They should because L @ 0 = 0
    mask_rot = torch.any(x_rot != 0.0, dim=-1)
    print(f"Mask sum after transform (should be 5): {mask_rot.sum(dim=1).tolist()}")
    
    if not torch.equal(mask, mask_rot):
        print("BUG FOUND: Mask changes after Lorentz transformation!")
        return False
    else:
        print("OK: Mask is preserved under transformation")
        return True


def test_gradient_flow():
    """Test that gradients flow correctly through the symmetry loss."""
    print("\n" + "="*60)
    print("TEST 7: Do gradients flow through symmetry loss?")
    print("="*60)
    
    model = DeepSets(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=4,
        num_rho_layers=4,
        pool_mode='sum',
    ).to(device)
    
    B, N = 32, 64
    x = torch.randn(B, N, 4).to(device)
    x[..., 0] = x[..., 0].abs() + 1
    
    # Compute symmetry loss and backpropagate
    loss = lorentz_orbit_variance_loss(model, x, layer_idx=1, std_eta=0.5)
    loss.backward()
    
    # Check if layer 1 weights have gradients
    layer1_weight_grad = model.phi_layers[0].weight.grad
    layer1_bias_grad = model.phi_layers[0].bias.grad
    
    print(f"Layer 1 weight grad norm: {layer1_weight_grad.norm().item():.6e}")
    print(f"Layer 1 bias grad norm: {layer1_bias_grad.norm().item():.6e}")
    
    if layer1_weight_grad.norm().item() < 1e-10:
        print("BUG FOUND: No gradients flowing to layer 1 weights!")
        return False
    else:
        print("OK: Gradients flow to layer 1")
        return True


def test_same_transform_gives_zero():
    """Test that applying the same transform twice gives zero loss."""
    print("\n" + "="*60)
    print("TEST 8: Does same transform give zero difference?")
    print("="*60)
    
    model = DeepSets(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=4,
        num_rho_layers=4,
        pool_mode='sum',
    ).to(device)
    
    B, N = 4, 10
    x = torch.randn(B, N, 4).to(device)
    x[..., 0] = x[..., 0].abs() + 1
    
    # Apply the same transform twice
    L = rand_lorentz(shape=torch.Size([B]), std_eta=0.5, device=device, dtype=torch.float32)
    x_rot = torch.matmul(L.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    with torch.no_grad():
        h1 = model.forward_with_intermediate(x_rot, layer_idx=1)
        h2 = model.forward_with_intermediate(x_rot, layer_idx=1)
    
    diff = (h1 - h2).abs().max().item()
    print(f"Diff between same-input activations: {diff:.6e}")
    
    if diff > 1e-6:
        print("WARNING: Same input gives different outputs!")
        return False
    else:
        print("OK: Same input gives same output")
        return True


def main():
    print("="*80)
    print("SYMMETRY LOSS BUG HUNTING")
    print("="*80)
    
    results = []
    results.append(("L1 != L2", test_lorentz_transforms_are_different()))
    results.append(("Mass preserved", test_lorentz_preserves_mass()))
    results.append(("m² gives zero loss", test_invariant_function_gives_zero_loss()))
    results.append(("Energy gives nonzero loss", test_non_invariant_function_gives_nonzero_loss()))
    results.append(("Random DeepSets nonzero", test_deepsets_layer1_non_invariance()))
    results.append(("Mask correct", test_mask_computation()))
    results.append(("Gradients flow", test_gradient_flow()))
    results.append(("Same input same output", test_same_transform_gives_zero()))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed - no obvious bugs in symmetry loss computation.")
        print("The behavior might be due to something else...")
    else:
        print("\nSome tests FAILED - there may be bugs to fix!")


if __name__ == '__main__':
    main()

