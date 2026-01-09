#!/usr/bin/env python3
"""
Analyze layer 1 weights to understand what representation is learned.

Examines:
1. Weight matrix structure (does it approximate Lorentz-invariant computations?)
2. Weight norms and distributions
3. Whether weights project onto invariant subspaces
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models import DeepSets

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path: str, num_phi_layers: int = 4, num_rho_layers: int = 4, 
               hidden_channels: int = 128, num_kps: int = 5) -> DeepSets:
    """Load a trained DeepSets model from checkpoint."""
    model = DeepSets(
        in_channels=4,
        out_channels=num_kps,
        hidden_channels=hidden_channels,
        num_phi_layers=num_phi_layers,
        num_rho_layers=num_rho_layers,
        pool_mode='sum',
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def analyze_layer1_weights(model, model_name: str):
    """
    Analyze the weights of the first phi layer.
    
    The first phi layer computes: h = W @ x + b
    where x = (E, px, py, pz) is a 4-vector.
    
    For Lorentz invariance, the only scalar we can construct from a single 4-vector
    is the mass squared: m² = E² - px² - py² - pz² = x^T @ g @ x
    where g = diag(1, -1, -1, -1) is the Minkowski metric.
    
    We analyze whether the learned weights approximate any invariant structure.
    """
    # Get layer 1 weights
    weight = model.phi_layers[0].weight.detach().cpu().numpy()  # (hidden, 4)
    bias = model.phi_layers[0].bias.detach().cpu().numpy()  # (hidden,)
    
    print(f"\n{'='*60}")
    print(f"Analysis: {model_name}")
    print(f"{'='*60}")
    print(f"Layer 1 weight shape: {weight.shape}")
    print(f"Layer 1 bias shape: {bias.shape}")
    
    # Basic statistics
    print(f"\nWeight Statistics:")
    print(f"  Mean:    {weight.mean():.6f}")
    print(f"  Std:     {weight.std():.6f}")
    print(f"  Min:     {weight.min():.6f}")
    print(f"  Max:     {weight.max():.6f}")
    print(f"  L2 norm: {np.linalg.norm(weight):.6f}")
    
    print(f"\nBias Statistics:")
    print(f"  Mean:    {bias.mean():.6f}")
    print(f"  Std:     {bias.std():.6f}")
    print(f"  Min:     {bias.min():.6f}")
    print(f"  Max:     {bias.max():.6f}")
    
    # Analyze weight columns (contribution of each input component)
    # Column 0: E, Column 1: px, Column 2: py, Column 3: pz
    col_norms = np.linalg.norm(weight, axis=0)
    print(f"\nColumn norms (input importance):")
    print(f"  E:  {col_norms[0]:.4f}")
    print(f"  px: {col_norms[1]:.4f}")
    print(f"  py: {col_norms[2]:.4f}")
    print(f"  pz: {col_norms[3]:.4f}")
    print(f"  Spatial (sqrt(px²+py²+pz²)): {np.sqrt(col_norms[1]**2 + col_norms[2]**2 + col_norms[3]**2):.4f}")
    
    # Check if any row approximates the Minkowski metric signature (1, -1, -1, -1)
    # This would indicate attempting to compute something like m² = E² - |p|²
    metric_pattern = np.array([1, -1, -1, -1])
    
    # Normalize rows and compute alignment with metric pattern
    row_norms = np.linalg.norm(weight, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)
    normalized_rows = weight / row_norms
    
    # Alignment with (1, -1, -1, -1) / ||metric||
    metric_normalized = metric_pattern / np.linalg.norm(metric_pattern)
    alignments = np.abs(normalized_rows @ metric_normalized)
    
    print(f"\nAlignment with Minkowski signature (1,-1,-1,-1):")
    print(f"  Max alignment:  {alignments.max():.4f} (row {alignments.argmax()})")
    print(f"  Mean alignment: {alignments.mean():.4f}")
    print(f"  Min alignment:  {alignments.min():.4f}")
    
    # Check how many rows have high alignment (>0.9)
    high_align_count = (alignments > 0.9).sum()
    print(f"  Rows with >0.9 alignment: {high_align_count}/{len(alignments)}")
    
    # Also check alignment with (1, 1, 1, 1) - this would be sum of components
    sum_pattern = np.array([1, 1, 1, 1]) / 2  # normalized
    sum_alignments = np.abs(normalized_rows @ sum_pattern)
    print(f"\nAlignment with sum pattern (1,1,1,1):")
    print(f"  Max alignment:  {sum_alignments.max():.4f}")
    print(f"  Mean alignment: {sum_alignments.mean():.4f}")
    
    # Check if rows are nearly zero (indicating collapse to constant)
    near_zero_rows = (row_norms.flatten() < 0.01).sum()
    print(f"\nRows with near-zero norm (<0.01): {near_zero_rows}/{weight.shape[0]}")
    
    # Analyze weight correlation structure
    weight_cov = np.corrcoef(weight.T)
    print(f"\nInput correlation structure:")
    print(f"  E-px correlation:  {weight_cov[0,1]:.4f}")
    print(f"  E-py correlation:  {weight_cov[0,2]:.4f}")
    print(f"  E-pz correlation:  {weight_cov[0,3]:.4f}")
    print(f"  px-py correlation: {weight_cov[1,2]:.4f}")
    print(f"  px-pz correlation: {weight_cov[1,3]:.4f}")
    print(f"  py-pz correlation: {weight_cov[2,3]:.4f}")
    
    return {
        'weight': weight,
        'bias': bias,
        'col_norms': col_norms,
        'metric_alignments': alignments,
        'row_norms': row_norms.flatten(),
    }


def plot_weight_comparison(results_dict: dict):
    """Plot comparison of weights across models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    model_names = list(results_dict.keys())
    
    # Plot 1: Weight matrices as heatmaps
    for idx, (name, result) in enumerate(results_dict.items()):
        ax = axes[0, idx]
        im = ax.imshow(result['weight'], aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax.set_title(f'{name}\nWeight matrix')
        ax.set_xlabel('Input (E, px, py, pz)')
        ax.set_ylabel('Hidden dim')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['E', 'px', 'py', 'pz'])
        plt.colorbar(im, ax=ax)
    
    # Plot 2: Column norms (input importance)
    ax = axes[1, 0]
    x = np.arange(4)
    width = 0.25
    for idx, (name, result) in enumerate(results_dict.items()):
        ax.bar(x + idx * width, result['col_norms'], width, label=name)
    ax.set_xlabel('Input component')
    ax.set_ylabel('Column L2 norm')
    ax.set_title('Input importance by model')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['E', 'px', 'py', 'pz'])
    ax.legend()
    
    # Plot 3: Minkowski alignment histogram
    ax = axes[1, 1]
    for name, result in results_dict.items():
        ax.hist(result['metric_alignments'], bins=30, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Alignment with (1,-1,-1,-1)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Minkowski alignment')
    ax.legend()
    
    # Plot 4: Row norm distribution
    ax = axes[1, 2]
    for name, result in results_dict.items():
        ax.hist(result['row_norms'], bins=30, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Row L2 norm')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of row norms')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('weight_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved weight analysis plot to weight_analysis.png")
    plt.show()


def main():
    print("="*80)
    print("LAYER 1 WEIGHT ANALYSIS")
    print("="*80)
    print("\nThe first phi layer computes: h = ReLU(W @ x + b)")
    print("where x = (E, px, py, pz) is a particle's 4-vector.")
    print("\nWe analyze whether the learned weights approximate Lorentz-invariant")
    print("computations, particularly the Minkowski metric signature (1, -1, -1, -1).")
    print("="*80)
    
    # Define models to analyze
    model_configs = [
        ('4x4_none.pt', 'Baseline'),
        ('4x4_layer1.pt', 'Layer1 Sym'),
        ('4x4_layer1_strong.pt', 'Strong Sym'),
    ]
    
    results = {}
    
    for model_path, model_name in model_configs:
        if not Path(model_path).exists():
            print(f"Skipping {model_name}: {model_path} not found")
            continue
            
        model = load_model(model_path)
        results[model_name] = analyze_layer1_weights(model, model_name)
    
    # Plot comparison
    if len(results) > 0:
        print("\n" + "="*80)
        print("GENERATING COMPARISON PLOTS")
        print("="*80)
        plot_weight_comparison(results)
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Key insights:
1. If weights have high Minkowski alignment → model learns mass-like features
2. If column norms are similar → model uses all input components equally
3. If row norms are small → model collapses to near-constant output
4. High E column norm vs spatial → model emphasizes energy over momenta
5. Symmetry training should push weights toward invariant structures
""")


if __name__ == '__main__':
    main()

