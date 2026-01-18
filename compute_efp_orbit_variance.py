#!/usr/bin/env python3
"""
Visualize KP/EFP orbit distributions under the augmentation-aware transformation scheme.

For a single event:
1. Sample rotation G and generate augmented data X_aug = G @ X (store G)
2. Compute original KP/EFPs from unaugmented data X
3. Sample many Lorentz transforms using L = G' @ G^(-1)
4. Compute KP/EFPs for each transformed version
5. Plot one histogram per KP/EFP (edgelist) showing distributions with original values marked

Use --measure kinematic for Lorentz-invariant kinematic polynomials (KPs).
Use --measure efp for non-invariant Energy Flow Polynomials (EFPs).

This shows how KP/EFPs vary under Lorentz transformations while keeping data in-distribution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from symmetry import rand_lorentz, lorentz_inverse
from data.kp_dataset import compute_kps
from train import load_efp_preset
import energyflow as ef


def compute_efp_orbits_for_events(
    n_particles: int = 32,
    n_transforms: int = 500,
    std_eta: float = 0.5,
    efp_preset: str = 'deg3',
    measure: str = 'efp',  # 'kinematic' for KP or 'efp' for EFP
    efp_beta: float = 2.0,
    efp_kappa: float = 1.0,
    efp_normed: bool = False,
    seed: int = 42,
):
    """
    Compute KP/EFP orbits for a single event under augmentation-aware transformations.
    
    1. Generate original data X
    2. Sample rotation G and apply augmentation: X_aug = G @ X (store G)
    3. Compute original KP/EFPs from unaugmented data
    4. For each of n_transforms:
       - Sample fresh G'
       - Compute L = G' @ G^(-1)
       - Apply: X_transformed = L @ X_aug = G' @ X
       - Compute KP/EFP(X_transformed)
    
    Args:
        measure: 'kinematic' for Lorentz-invariant KPs or 'efp' for non-invariant EFPs
    
    Returns:
        Dictionary with:
        - efp_original: (num_kps,) original KP/EFP values computed from X
        - efp_orbits: (n_transforms, num_kps) KP/EFP values for each transform
        - edges_list: Edge configurations
        - measure: The measure used
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load edge configurations
    edges_list = load_efp_preset(efp_preset, 'config')
    num_kps = len(edges_list)
    
    # Determine measure parameters
    if measure == 'kinematic':
        measure_name = 'kinematic'
        beta, kappa, normed = 2.0, 1.0, False  # Ignored by kinematic measure
        measure_label = 'KP'
    elif measure == 'efp':
        measure_name = 'eeefm'
        beta, kappa, normed = efp_beta, efp_kappa, efp_normed
        measure_label = 'EFP'
    else:
        raise ValueError(f"Unknown measure: {measure}. Must be 'kinematic' or 'efp'")
    
    print(f"Computing {measure_label} orbits")
    print(f"  Particles: {n_particles}")
    print(f"  Transforms: {n_transforms}")
    print(f"  std_eta: {std_eta}")
    print(f"  Preset: {efp_preset} ({num_kps} {measure_label}s)")
    if measure == 'efp':
        print(f"  EFP params: beta={beta}, kappa={kappa}, normed={normed}")
    print()
    
    # Generate original data for 1 event
    X = ef.gen_random_events_mcom(1, n_particles, dim=4).astype(np.float32)
    
    # Sample rotation G
    G = rand_lorentz(
        shape=torch.Size([1]),
        std_eta=std_eta,
        device='cpu',
        dtype=torch.float32,
    )
    
    # Apply augmentation: X_aug = G @ X
    X_torch = torch.from_numpy(X)
    X_aug = torch.einsum('nij,nmj->nmi', G, X_torch).numpy()
    
    # Compute original KP/EFPs from the original unaugmented data X
    kp_original = compute_kps(X, edges_list, measure=measure_name, 
                              beta=beta, kappa=kappa, normed=normed)
    kp_original = kp_original[0]  # Flatten to (num_kps,)
    
    # Compute G^(-1)
    G_inv = lorentz_inverse(G)
    
    # Storage for orbit samples
    # Shape: (n_transforms, num_kps)
    kp_orbits = np.zeros((n_transforms, num_kps), dtype=np.float32)
    
    print("Sampling orbits...")
    for i in tqdm(range(n_transforms)):
        # Sample fresh G'
        G_prime = rand_lorentz(
            shape=torch.Size([1]),
            std_eta=std_eta,
            device='cpu',
            dtype=torch.float32,
        )
        
        # Augmentation-aware method: L = G' @ G^(-1)
        L = torch.bmm(G_prime, G_inv)
        
        # Apply L to X_aug: L @ X_aug = G' @ X
        X_aug_torch = torch.from_numpy(X_aug)
        X_transformed = torch.einsum('nij,nmj->nmi', L, X_aug_torch).numpy()
        
        # Compute KP/EFPs on augmentations
        kp_orbits[i, :] = compute_kps(
            X_transformed, edges_list, measure=measure_name,
            beta=beta, kappa=kappa, normed=normed
        )[0]  # Flatten to (num_kps,)
    
    # Compute relative symmetry penalty for a perfect regressor
    # This is: ||a - b||^2 / (||a||^2 + ||b||^2 + eps) averaged over pairs
    # For each EFP, sample pairs from the orbit and compute the penalty
    eps = 1e-8
    n_pairs = min(1000, n_transforms * (n_transforms - 1) // 2)
    rel_sym_penalty = np.zeros(num_kps, dtype=np.float32)
    
    for kp_idx in range(num_kps):
        orbit_vals = kp_orbits[:, kp_idx]
        # Sample random pairs
        penalties = []
        for _ in range(n_pairs):
            i, j = np.random.choice(n_transforms, 2, replace=False)
            a, b = orbit_vals[i], orbit_vals[j]
            penalty = (a - b)**2 / (a**2 + b**2 + eps)
            penalties.append(penalty)
        rel_sym_penalty[kp_idx] = np.mean(penalties)
    
    results = {
        'edges_list': edges_list,
        'num_efps': num_kps,  # Keep name for backward compatibility
        'n_transforms': n_transforms,
        'efp_original': kp_original,  # Keep name for backward compatibility
        'efp_orbits': kp_orbits,      # Keep name for backward compatibility
        'measure': measure,
        'measure_label': measure_label,
        'rel_sym_penalty': rel_sym_penalty,  # Relative symmetry penalty per KP/EFP
    }
    
    return results




def plot_efp_orbits(results: dict, save_path: str = None):
    """
    Plot histograms of EFP orbit distributions for each EFP.
    
    For each EFP (edgelist), shows:
    - Histogram of EFP values across all transforms
    - Vertical line marking the original EFP value from unaugmented data
    - Relative symmetry penalty for a perfect regressor (only for EFPs)
    - Uses log1p scale on x-axis
    """
    num_efps = results['num_efps']
    efp_original = results['efp_original']  # (num_efps,)
    efp_orbits = results['efp_orbits']      # (n_transforms, num_efps)
    edges_list = results['edges_list']
    rel_sym_penalty = results.get('rel_sym_penalty', None)
    measure = results.get('measure', 'efp')
    
    # Create figure with one subplot per EFP (edgelist)
    fig, axes = plt.subplots(1, num_efps, figsize=(5*num_efps, 5))
    
    # Handle case where num_efps == 1 (axes would be 1D instead of array)
    if num_efps == 1:
        axes = [axes]
    
    for efp_idx in range(num_efps):
        ax = axes[efp_idx]
        
        # Get orbit values for this EFP
        orbit_values = efp_orbits[:, efp_idx]  # (n_transforms,)
        
        # Get original value for this EFP
        original_value = efp_original[efp_idx]  # Single value
        
        # Apply log1p transformation for plotting
        orbit_log1p = np.log1p(orbit_values)
        original_log1p = np.log1p(original_value)
        
        # Create histogram
        bins = np.linspace(orbit_log1p.min(), orbit_log1p.max(), 50)
        ax.hist(orbit_log1p, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Draw vertical line for the original KP/EFP value
        measure_label = results.get('measure_label', 'EFP')
        ax.axvline(original_log1p, color='red', linestyle='--', 
                  linewidth=2, alpha=0.8, label=f'Original {measure_label}')
        
        # Add relative symmetry penalty text box (only for EFPs, not KPs)
        if rel_sym_penalty is not None and measure == 'efp':
            penalty = rel_sym_penalty[efp_idx]
            textstr = f'Rel. Sym. Penalty\n(perfect regressor):\n{penalty:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=props)
        
        # Labels and title
        edges_str = str(edges_list[efp_idx])
        if len(edges_str) > 40:
            edges_str = edges_str[:37] + '...'
        measure_label = results.get('measure_label', 'EFP')
        ax.set_xlabel(f'log1p({measure_label} Value)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{measure_label} {efp_idx+1}\n{edges_str}', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    measure_label = results.get('measure_label', 'EFP')
    plt.suptitle(f'{measure_label} Orbit Distributions Under Lorentz Transformations\n'
                 f'({results["n_transforms"]} transforms)',
                 fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize KP/EFP orbit distributions under augmentation scheme'
    )
    parser.add_argument('--measure', type=str, default='efp',
                        choices=['kinematic', 'efp'],
                        help='Measure type: "kinematic" for Lorentz-invariant KPs, "efp" for non-invariant EFPs')
    parser.add_argument('--n-particles', type=int, default=32,
                        help='Particles per event')
    parser.add_argument('--n-transforms', type=int, default=500,
                        help='Number of Lorentz transforms per event')
    parser.add_argument('--std-eta', type=float, default=0.5,
                        help='Rapidity std for transformations')
    parser.add_argument('--efp-preset', type=str, default='deg3',
                        help='EFP/KP preset name')
    parser.add_argument('--efp-beta', type=float, default=2.0,
                        help='EFP angular weighting (only used with --measure efp)')
    parser.add_argument('--efp-kappa', type=float, default=1.0,
                        help='EFP energy weighting (only used with --measure efp)')
    parser.add_argument('--efp-normed', action='store_true',
                        help='Normalize EFP energies (only used with --measure efp)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save plot (default: auto-generated based on measure)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    
    args = parser.parse_args()
    
    # Auto-generate plot filename if not provided
    if args.save_plot is None:
        measure_label = 'kp' if args.measure == 'kinematic' else 'efp'
        args.save_plot = f'{measure_label}_orbit_distributions.png'
    
    results = compute_efp_orbits_for_events(
        n_particles=args.n_particles,
        n_transforms=args.n_transforms,
        std_eta=args.std_eta,
        efp_preset=args.efp_preset,
        measure=args.measure,
        efp_beta=args.efp_beta,
        efp_kappa=args.efp_kappa,
        efp_normed=args.efp_normed,
        seed=args.seed,
    )
    
    measure_label = results['measure_label']
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nOriginal {measure_label} values:")
    print(f"{measure_label:<8} {'Value':<15}")
    print("-" * 70)
    for efp_idx in range(results['num_efps']):
        print(f"{efp_idx+1:<8} {results['efp_original'][efp_idx]:<15.4e}")
    
    print(f"\nOrbit statistics (across {results['n_transforms']} transforms):")
    print(f"{measure_label:<8} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 70)
    for efp_idx in range(results['num_efps']):
        orbit_values = results['efp_orbits'][:, efp_idx]
        print(f"{efp_idx+1:<8} {orbit_values.mean():<15.4e} {orbit_values.std():<15.4e} "
              f"{orbit_values.min():<15.4e} {orbit_values.max():<15.4e}")
    
    # Print relative symmetry penalty (only for EFPs)
    if results['measure'] == 'efp' and 'rel_sym_penalty' in results:
        print(f"\nRelative symmetry penalty (perfect {measure_label} regressor):")
        print(f"  This is ||a-b||^2 / (||a||^2 + ||b||^2) averaged over pairs of orbit samples.")
        print(f"  A perfect EFP regressor would incur this penalty when used with symmetry loss.")
        print(f"{measure_label:<8} {'Rel. Sym. Penalty':<20}")
        print("-" * 70)
        for efp_idx in range(results['num_efps']):
            print(f"{efp_idx+1:<8} {results['rel_sym_penalty'][efp_idx]:<20.6f}")
    
    if not args.no_plot:
        plot_efp_orbits(results, save_path=args.save_plot)


if __name__ == '__main__':
    main()
