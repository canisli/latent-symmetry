#!/usr/bin/env python3
"""
Check if the model has learned to saturate ReLU (produce negative pre-ReLU values
that become 0 after ReLU).

This could explain why outputs are invariant despite non-zero weights:
if most neurons output 0, the output is effectively constant!
"""

import torch
import numpy as np
from pathlib import Path
import energyflow as ef
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


class DeepSetsWithPreReLU(DeepSets):
    """Modified DeepSets that can return pre-ReLU activations."""
    
    def forward_pre_relu(self, inputs, layer_idx):
        """Return activations BEFORE ReLU at specified layer."""
        h = inputs
        for i, layer in enumerate(self.phi_layers, start=1):
            h_pre = layer(h)  # Before ReLU
            if i == layer_idx:
                return h_pre
            h = self.phi_act(h_pre)  # After ReLU
        raise ValueError(f"Layer {layer_idx} not found in phi layers")


def analyze_relu_saturation(model_path: str, model_name: str):
    """Analyze ReLU saturation in layer 1."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    model = load_model(model_path)
    
    # Copy weights to our modified model
    modified_model = DeepSetsWithPreReLU(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=4,
        num_rho_layers=4,
        pool_mode='sum',
    ).to(device)
    modified_model.load_state_dict(model.state_dict())
    modified_model.eval()
    
    # Generate test data
    n_events = 100
    n_particles = 128
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    X = torch.from_numpy(X).to(device)
    
    with torch.no_grad():
        # Get pre-ReLU and post-ReLU activations for layer 1
        h_pre = modified_model.forward_pre_relu(X, layer_idx=1)  # (B, N, 128)
        h_post = model.forward_with_intermediate(X, layer_idx=1)  # (B, N, 128)
    
    h_pre_np = h_pre.cpu().numpy()
    h_post_np = h_post.cpu().numpy()
    
    # Analyze pre-ReLU distribution
    print(f"\nPre-ReLU (before activation):")
    print(f"  Mean: {h_pre_np.mean():.4f}")
    print(f"  Std:  {h_pre_np.std():.4f}")
    print(f"  Min:  {h_pre_np.min():.4f}")
    print(f"  Max:  {h_pre_np.max():.4f}")
    
    # Fraction of negative (dead) neurons
    fraction_negative = (h_pre_np < 0).mean() * 100
    print(f"  Fraction negative (→0 after ReLU): {fraction_negative:.1f}%")
    
    # Per-neuron analysis: how often is each neuron dead?
    neuron_dead_fraction = (h_pre_np < 0).mean(axis=(0, 1))  # (128,)
    mostly_dead = (neuron_dead_fraction > 0.9).sum()
    always_dead = (neuron_dead_fraction > 0.99).sum()
    
    print(f"  Neurons >90% dead: {mostly_dead}/128")
    print(f"  Neurons >99% dead: {always_dead}/128")
    
    # Analyze post-ReLU distribution
    print(f"\nPost-ReLU (after activation):")
    print(f"  Mean: {h_post_np.mean():.4f}")
    print(f"  Std:  {h_post_np.std():.4f}")
    print(f"  Fraction zero: {(h_post_np == 0).mean() * 100:.1f}%")
    
    # If output is mostly zeros, that explains invariance!
    if fraction_negative > 80:
        print("\n  *** HIGH ReLU SATURATION - most neurons output 0 ***")
        print("  This could explain apparent invariance!")
    elif fraction_negative > 50:
        print("\n  Moderate ReLU saturation")
    else:
        print("\n  Low ReLU saturation - neurons are active")
    
    return fraction_negative, neuron_dead_fraction


def main():
    print("="*80)
    print("ReLU SATURATION ANALYSIS")
    print("="*80)
    print("\nIf most pre-ReLU values are negative, they become 0 after ReLU.")
    print("This would make the layer output nearly constant (zero)!")
    
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
        
        frac_neg, dead_frac = analyze_relu_saturation(model_path, model_name)
        results.append((model_name, frac_neg))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, frac in results:
        print(f"  {name}: {frac:.1f}% negative (dead)")
    
    if len(results) >= 2:
        if results[1][1] > results[0][1] + 20:
            print("\n*** FINDING: Symmetry training increases ReLU saturation! ***")
            print("This could explain how outputs become invariant:")
            print("Most neurons die → output becomes ~0 → invariant!")


if __name__ == '__main__':
    main()

