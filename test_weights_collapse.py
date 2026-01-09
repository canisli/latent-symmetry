#!/usr/bin/env python3
"""
Test if symmetry training is actually driving layer 1 weights toward zero.

If the symmetry loss is working correctly, layer 1 weights should collapse
to near-zero because there is NO non-trivial per-particle Lorentz invariant
that a linear layer can compute (since particles are massless).
"""

import torch
import numpy as np
from pathlib import Path
from models import DeepSets

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path: str, num_phi_layers: int = 4, num_rho_layers: int = 4) -> DeepSets:
    model = DeepSets(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=num_phi_layers,
        num_rho_layers=num_rho_layers,
        pool_mode='sum',
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def analyze_weight_collapse(model, model_name):
    """Check if weights have collapsed to near-zero."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    weight = model.phi_layers[0].weight.detach().cpu().numpy()  # (128, 4)
    bias = model.phi_layers[0].bias.detach().cpu().numpy()
    
    # Check weight statistics
    weight_norm = np.linalg.norm(weight)
    weight_max = np.abs(weight).max()
    weight_mean = np.abs(weight).mean()
    
    print(f"Layer 1 weight:")
    print(f"  Frobenius norm: {weight_norm:.4f}")
    print(f"  Max absolute:   {weight_max:.4f}")
    print(f"  Mean absolute:  {weight_mean:.4f}")
    
    # Check per-neuron weight norms
    row_norms = np.linalg.norm(weight, axis=1)
    print(f"  Row norms - min: {row_norms.min():.4f}, max: {row_norms.max():.4f}")
    print(f"  Rows with norm < 0.01: {(row_norms < 0.01).sum()}/{len(row_norms)}")
    print(f"  Rows with norm < 0.1:  {(row_norms < 0.1).sum()}/{len(row_norms)}")
    
    # Compare to random initialization
    # Xavier init for (128, 4): std ≈ sqrt(2/(128+4)) ≈ 0.123
    random_expected_norm = 0.123 * np.sqrt(128 * 4)
    print(f"  Expected random init norm: {random_expected_norm:.4f}")
    print(f"  Ratio vs random: {weight_norm / random_expected_norm:.4f}")
    
    # Check bias
    bias_norm = np.linalg.norm(bias)
    print(f"\nLayer 1 bias:")
    print(f"  Norm: {bias_norm:.4f}")
    print(f"  Mean: {bias.mean():.4f}, Std: {bias.std():.4f}")
    
    # KEY QUESTION: Are weights small enough that layer 1 outputs are
    # dominated by biases?
    # For input x with ||x|| ~ 1, output h = W@x + b
    # If ||W|| is small, then ||W@x|| << ||b||
    
    # Typical input magnitude (after scaling)
    input_scale = 1.0  # Approximately unit scale after preprocessing
    expected_wx_norm = weight_norm * input_scale / np.sqrt(4)  # Rough estimate
    
    print(f"\nKey comparison:")
    print(f"  Expected ||W@x||: {expected_wx_norm:.4f}")
    print(f"  ||b||:            {bias_norm:.4f}")
    print(f"  Ratio W@x / b:    {expected_wx_norm / (bias_norm + 1e-10):.4f}")
    
    if expected_wx_norm / (bias_norm + 1e-10) < 0.1:
        print("  → Weights are SMALL: output dominated by bias (collapsed)")
    else:
        print("  → Weights are SIGNIFICANT: output has non-trivial x dependence")
    
    return weight_norm, bias_norm


def test_layer1_output_variance():
    """Test if layer 1 outputs are nearly constant (as they should be if collapsed)."""
    print(f"\n{'='*60}")
    print("TEST: Is layer 1 output variance low?")
    print(f"{'='*60}")
    
    import energyflow as ef
    
    n_events = 100
    n_particles = 128
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    X = torch.from_numpy(X).to(device)
    
    model_configs = [
        ('4x4_none.pt', 'Baseline'),
        ('4x4_layer1.pt', 'Layer1 Sym'),
        ('4x4_layer1_strong.pt', 'Strong Sym'),
    ]
    
    for model_path, model_name in model_configs:
        if not Path(model_path).exists():
            continue
            
        model = load_model(model_path)
        model.eval()
        
        with torch.no_grad():
            h1 = model.forward_with_intermediate(X, layer_idx=1)  # (B, N, 128)
        
        # Check variance across particles and events
        h1_np = h1.cpu().numpy()
        
        # Per-event variance (average over particles)
        per_event_var = h1_np.var(axis=1).mean()
        
        # Per-particle variance (average over events)
        per_particle_var = h1_np.var(axis=0).mean()
        
        # Total variance
        total_var = h1_np.var()
        
        # Mean activation (should be ~bias if weights collapsed)
        mean_activation = h1_np.mean()
        
        print(f"\n{model_name}:")
        print(f"  Mean activation:     {mean_activation:.4f}")
        print(f"  Total variance:      {total_var:.4f}")
        print(f"  Per-event variance:  {per_event_var:.4f}")
        print(f"  Per-particle var:    {per_particle_var:.4f}")
        
        # If weights collapsed, variance should be near zero
        if total_var < 0.01:
            print("  → Layer 1 outputs are nearly CONSTANT")
        elif total_var < 0.1:
            print("  → Layer 1 outputs have LOW variance")
        else:
            print("  → Layer 1 outputs have SIGNIFICANT variance")


def main():
    print("="*80)
    print("WEIGHT COLLAPSE ANALYSIS")
    print("="*80)
    print("\nIf layer 1 symmetry is truly enforced, weights should collapse")
    print("to near-zero (since there's no invariant a linear layer can compute).")
    
    model_configs = [
        ('4x4_none.pt', 'Baseline'),
        ('4x4_layer1.pt', 'Layer1 Sym'),
        ('4x4_layer1_strong.pt', 'Strong Sym'),
    ]
    
    results = []
    for model_path, model_name in model_configs:
        if not Path(model_path).exists():
            print(f"Skipping {model_name}: {model_path} not found")
            continue
            
        model = load_model(model_path)
        w_norm, b_norm = analyze_weight_collapse(model, model_name)
        results.append((model_name, w_norm, b_norm))
    
    test_layer1_output_variance()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        baseline_norm = results[0][1]
        for name, w_norm, b_norm in results[1:]:
            ratio = w_norm / baseline_norm
            print(f"{name}: weight norm = {ratio*100:.1f}% of baseline")
            
        print(f"\nIf symmetry constraint is working:")
        print("  - Weight norm should be << baseline (e.g., <10%)")
        print("  - Layer 1 output variance should be very low")
        print("  - BUT model still achieves good task loss")
        print(f"\nThis would indicate a BUG because with collapsed weights,")
        print("the model cannot compute non-trivial functions of the input!")


if __name__ == '__main__':
    main()

