#!/usr/bin/env python3
"""
Fixed symmetry loss that accounts for ReLU saturation.

The original symmetry loss can be "cheated" by killing neurons (pushing pre-ReLU 
values negative so they become 0 after ReLU). This gives trivial invariance.

This fixed version only considers ACTIVE neurons when computing the loss.
"""

import torch
from symmetry import rand_lorentz


def lorentz_orbit_variance_loss_fixed(
    model,
    x: torch.Tensor,
    layer_idx: int,
    std_eta: float = 0.5,
    n_max_std_eta: float = 3.0,
    generator: torch.Generator = None,
    mask: torch.Tensor = None,
    activity_threshold: float = 1e-6,  # NEW: threshold for "active" neurons
):
    """
    Fixed symmetry loss that only considers active (non-zero) neurons.
    
    The original implementation was fooled by ReLU saturation:
    - Neurons outputting 0 are trivially invariant
    - Model learns to kill neurons rather than compute invariants
    
    This version:
    - Identifies which neurons are active in BOTH transforms
    - Only computes loss on those active neurons
    - This forces the model to learn true invariance for active features
    
    Args:
        model: Model with forward_with_intermediate() method
        x: Input 4-vectors of shape (batch_size, num_particles, 4)
        layer_idx: Which layer's activations to compare (-1 for output)
        std_eta: Standard deviation of rapidity for boosts
        n_max_std_eta: Maximum number of standard deviations for truncation
        generator: Optional random generator for reproducibility
        mask: Optional boolean mask for particles
        activity_threshold: Threshold for considering a neuron "active"
    
    Returns:
        Scalar symmetry loss computed only over active neurons
    """
    B, N, D = x.shape
    assert D == 4, f"Expected 4-vectors, got dimension {D}"
    device, dtype = x.device, x.dtype
    
    # Sample two random Lorentz transformations
    L1 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    L2 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    
    # Apply Lorentz transformations to all particles
    x_rot1 = torch.matmul(L1.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.matmul(L2.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    # Get intermediate activations
    h1 = model.forward_with_intermediate(x_rot1, layer_idx, mask=mask)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx, mask=mask)
    
    # Handle per-particle representations (pre-pooling layers)
    if h1.dim() == 3:  # (B, N, hidden)
        if mask is None:
            mask = torch.any(x != 0.0, dim=-1)  # (B, N)
        
        # FIXED: Only consider neurons that are active in BOTH h1 and h2
        # Active means |value| > threshold
        active_h1 = h1.abs() > activity_threshold  # (B, N, hidden)
        active_h2 = h2.abs() > activity_threshold  # (B, N, hidden)
        active_mask = active_h1 | active_h2  # Active in either transform
        
        # Compute squared differences
        diff = h1 - h2
        per_element_loss = diff.pow(2)  # (B, N, hidden)
        
        # Apply both particle mask and activity mask
        particle_mask = mask.float().unsqueeze(-1)  # (B, N, 1)
        full_mask = particle_mask * active_mask.float()  # (B, N, hidden)
        
        # Average only over active elements
        total_active = full_mask.sum().clamp(min=1.0)
        loss = 0.5 * (per_element_loss * full_mask).sum() / total_active
        
        # Also return diagnostics
        fraction_active = active_mask.float().mean()
        
        return loss, fraction_active
    
    # Post-pooling layers: h1 and h2 have shape (B, hidden)
    active_h1 = h1.abs() > activity_threshold
    active_h2 = h2.abs() > activity_threshold
    active_mask = active_h1 | active_h2
    
    diff = h1 - h2
    per_element_loss = diff.pow(2)
    
    total_active = active_mask.float().sum().clamp(min=1.0)
    loss = 0.5 * (per_element_loss * active_mask.float()).sum() / total_active
    
    fraction_active = active_mask.float().mean()
    
    return loss, fraction_active


def test_fixed_loss():
    """Compare original and fixed loss on trained models."""
    import numpy as np
    from pathlib import Path
    import energyflow as ef
    from models import DeepSets
    from symmetry import lorentz_orbit_variance_loss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(model_path):
        model = DeepSets(
            in_channels=4, out_channels=5, hidden_channels=128,
            num_phi_layers=4, num_rho_layers=4, pool_mode='sum',
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    
    # Generate test data
    X = ef.gen_random_events_mcom(100, 128, dim=4).astype(np.float32)
    X = torch.from_numpy(X).to(device)
    
    print("="*80)
    print("COMPARING ORIGINAL vs FIXED SYMMETRY LOSS")
    print("="*80)
    
    model_configs = [
        ('4x4_none.pt', 'Baseline'),
        ('4x4_layer1.pt', 'Layer1 Sym'),
        ('4x4_layer1_strong.pt', 'Strong Sym'),
    ]
    
    for model_path, model_name in model_configs:
        if not Path(model_path).exists():
            continue
            
        model = load_model(model_path)
        
        with torch.no_grad():
            # Original loss
            orig_loss = lorentz_orbit_variance_loss(model, X, layer_idx=1, std_eta=0.5)
            
            # Fixed loss
            fixed_loss, frac_active = lorentz_orbit_variance_loss_fixed(model, X, layer_idx=1, std_eta=0.5)
        
        print(f"\n{model_name}:")
        print(f"  Original loss: {orig_loss.item():.4e}")
        print(f"  Fixed loss:    {fixed_loss.item():.4e}")
        print(f"  Ratio (fixed/orig): {fixed_loss.item() / (orig_loss.item() + 1e-10):.2f}x")
        print(f"  Fraction active neurons: {frac_active.item()*100:.1f}%")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("="*80)
    print("""
If the fixed loss is MUCH HIGHER than original for symmetry-trained models,
it confirms the model is "cheating" by killing neurons.

The original loss is fooled by dead neurons (0 = invariant).
The fixed loss only considers active neurons, revealing true non-invariance.
""")


if __name__ == '__main__':
    test_fixed_loss()

