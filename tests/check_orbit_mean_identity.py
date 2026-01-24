#!/usr/bin/env python
"""
Check whether m(x) ≈ μ for most x, explaining why Q=1 after the first layer.

m(x) = E_g[h(gx)]  (orbit mean for point x)
μ = E_x[h(x)]       (global sample mean)

If m(x) ≈ μ for all x, then Q = 1 - E[||m(x) - μ||²]/E[||h(x) - μ||²] ≈ 1.
"""

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

from latsym.models import MLP
from latsym.groups.so2 import rotate, sample_rotations


def main():
    # torch.manual_seed(42)
    
    # Create rotation-invariant data (uniform on annulus)
    N = 100000
    r_inner, r_outer = 0.0, 1.0
    r = torch.sqrt(torch.rand(N) * (r_outer**2 - r_inner**2) + r_inner**2)
    theta = torch.rand(N) * 2 * np.pi
    data = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    
    # Random MLP
    model = MLP([2, 128, 128, 128, 1])
    model.eval()
    
    # Get activations h(x) at layer 1
    with torch.no_grad():
        h = model.forward_with_intermediate(data, layer_idx=1)  # (N, 128)
    
    # Global mean μ = E_x[h(x)]
    mu = h.mean(dim=0)  # (128,)
    
    # Compute orbit mean m(x) = E_g[h(gx)] for each x
    n_orbit_samples = 1000
    orbit_sum = torch.zeros_like(h)
    for _ in range(n_orbit_samples):
        theta_g = sample_rotations(N)
        x_rot = rotate(data, theta_g)
        with torch.no_grad():
            h_rot = model.forward_with_intermediate(x_rot, layer_idx=1)
        orbit_sum += h_rot
    m = orbit_sum / n_orbit_samples  # (N, 128)
    
    # Pick a single neuron to visualize
    neuron_idx = 0
    m_neuron = m[:, neuron_idx]      # orbit mean for neuron 0: m_0(x) for each x
    h_neuron = h[:, neuron_idx]      # raw activation for neuron 0: h_0(x)
    mu_neuron = mu[neuron_idx]       # global mean for neuron 0: μ_0
    
    # Plot: histogram of m(x) for one neuron with line at μ
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(m_neuron.numpy(), bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(mu_neuron.item(), color='red', linewidth=2.5, 
               label=f'μ = {mu_neuron.item():.3f}')
    
    ax.set_xlabel('m(x) = E_g[h(gx)]  (orbit mean)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Histogram of orbit means m(x) for neuron {neuron_idx}\n'
                 f'(Layer 1, random init)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('orbit_mean_histogram.png', dpi=150)
    print(f"Saved to orbit_mean_histogram.png")
    
    # Statistics for single neuron
    print(f"\nNeuron {neuron_idx}:")
    print(f"  μ (global mean)     = {mu_neuron.item():.4f}")
    print(f"  E[m(x)]             = {m_neuron.mean().item():.4f}")
    print(f"  Std[m(x)]           = {m_neuron.std().item():.4f}")
    print(f"  Std[h(x)]           = {h_neuron.std().item():.4f}")
    
    # Compute Q using the identity (full D-dimensional)
    # Q = 1 - E[||m(x) - μ||²] / E[||h(x) - μ||²]
    numerator = ((m - mu) ** 2).sum(dim=1).mean()   # E[||m(x) - μ||²]
    denominator = ((h - mu) ** 2).sum(dim=1).mean() # E[||h(x) - μ||²]
    Q = 1 - numerator / denominator
    
    print(f"\nFull representation (D=128):")
    print(f"  E[||m(x) - μ||²] = {numerator.item():.4f}")
    print(f"  E[||h(x) - μ||²] = {denominator.item():.4f}")
    print(f"  Q = 1 - ratio    = {Q.item():.4f}")
    print(f"\nm(x) ≈ μ for all x → Q ≈ 1")

    plt.show()


if __name__ == "__main__":
    main()
