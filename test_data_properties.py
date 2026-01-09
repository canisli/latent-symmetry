#!/usr/bin/env python3
"""
Check properties of the generated data that might explain the results.

Potential issues:
1. Are the particles massless? If so, what invariants are available?
2. Is there structure in the data that makes invariance "easy"?
3. Are the KP targets actually varying, or nearly constant?
"""

import torch
import numpy as np
from pathlib import Path
import energyflow as ef
from data.kp_dataset import compute_kps, make_kp_dataloader
from train import load_efp_preset

def test_particle_masses():
    """Check if generated particles are massless."""
    print("\n" + "="*60)
    print("TEST: Particle mass distribution")
    print("="*60)
    
    # Generate events like in training
    n_events = 1000
    n_particles = 128
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    # Compute mass squared for each particle
    # m² = E² - px² - py² - pz²
    E = X[..., 0]
    px = X[..., 1]
    py = X[..., 2]
    pz = X[..., 3]
    
    m2 = E**2 - px**2 - py**2 - pz**2
    
    print(f"Mass squared statistics:")
    print(f"  Mean: {m2.mean():.6e}")
    print(f"  Std:  {m2.std():.6e}")
    print(f"  Min:  {m2.min():.6e}")
    print(f"  Max:  {m2.max():.6e}")
    
    # Check how many are effectively massless
    effectively_massless = (np.abs(m2) < 1e-6).mean() * 100
    print(f"  Effectively massless (|m²|<1e-6): {effectively_massless:.1f}%")
    
    if effectively_massless > 99:
        print("\n  IMPORTANT: Particles are massless!")
        print("  For massless particles, the only per-particle Lorentz invariant is m²=0")
        print("  This means layer 1 has NO non-trivial invariant to compute!")
    
    return m2


def test_kp_target_distribution():
    """Check the distribution of KP targets."""
    print("\n" + "="*60)
    print("TEST: KP target distribution")
    print("="*60)
    
    config_dir = Path(__file__).parent / "config"
    edges_list = load_efp_preset('deg3', str(config_dir))
    
    n_events = 1000
    n_particles = 128
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    Y = compute_kps(X, edges_list, measure='kinematic', coords='epxpypz')
    Y_log = np.log1p(Y)
    
    print(f"\nRaw KP statistics:")
    for i, edges in enumerate(edges_list):
        print(f"  KP{i+1} {edges}:")
        print(f"    Mean: {Y[:, i].mean():.4e}, Std: {Y[:, i].std():.4e}")
        print(f"    Min:  {Y[:, i].min():.4e}, Max: {Y[:, i].max():.4e}")
    
    print(f"\nLog1p(KP) statistics (actual training targets):")
    for i, edges in enumerate(edges_list):
        print(f"  KP{i+1}: Mean={Y_log[:, i].mean():.4f}, Std={Y_log[:, i].std():.4f}")
    
    # Check coefficient of variation (std/mean)
    print(f"\nCoefficient of variation (std/mean) for log1p targets:")
    for i in range(len(edges_list)):
        cv = Y_log[:, i].std() / np.abs(Y_log[:, i].mean())
        print(f"  KP{i+1}: {cv:.4f}")
    
    return Y, Y_log


def test_what_linear_layer_can_compute():
    """
    Test what a single linear layer can compute from a 4-vector.
    
    For a linear layer h = W @ x + b where x = (E, px, py, pz):
    - Each output h_i = w_i · x + b_i = w_i0*E + w_i1*px + w_i2*py + w_i3*pz + b_i
    
    For Lorentz invariance, we need h(L@x) = h(x) for all Lorentz L.
    This requires w_i · (L@x) = w_i · x, i.e., w_i^T L = w_i^T for all L.
    
    The only vectors satisfying this are w_i = 0 (since L can be any Lorentz transform).
    So a linear layer can only output constants (the biases) if we require Lorentz invariance.
    """
    print("\n" + "="*60)
    print("TEST: What can a linear layer compute that's invariant?")
    print("="*60)
    
    from symmetry import rand_lorentz
    
    # Create a random linear layer
    W = torch.randn(10, 4)
    b = torch.randn(10)
    
    # Create some test 4-vectors
    x = torch.randn(100, 4)
    x[:, 0] = x[:, 0].abs() + 1  # Positive energy
    
    # Compute h = W @ x^T + b for original x
    h_orig = (W @ x.T).T + b  # (100, 10)
    
    # Apply Lorentz transform and compute h
    L = rand_lorentz(shape=torch.Size([1]), std_eta=0.5)[0]
    x_rot = (L @ x.unsqueeze(-1)).squeeze(-1)
    h_rot = (W @ x_rot.T).T + b
    
    diff = (h_orig - h_rot).abs().mean().item()
    print(f"Mean absolute difference |h(x) - h(Lx)|: {diff:.4f}")
    print("(Should be large for random W, showing linear layers are NOT invariant)")
    
    # Now test with W = 0 (only biases)
    W_zero = torch.zeros(10, 4)
    h_orig_zero = (W_zero @ x.T).T + b
    h_rot_zero = (W_zero @ x_rot.T).T + b
    diff_zero = (h_orig_zero - h_rot_zero).abs().mean().item()
    print(f"With W=0: {diff_zero:.4e} (should be ~0)")
    
    return diff


def test_data_correlation():
    """Check if there's correlation structure in the data."""
    print("\n" + "="*60)
    print("TEST: Data correlation structure")
    print("="*60)
    
    n_events = 1000
    n_particles = 128
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    # Flatten to (n_events * n_particles, 4)
    X_flat = X.reshape(-1, 4)
    
    # Compute correlation matrix
    corr = np.corrcoef(X_flat.T)
    
    print("Correlation matrix of (E, px, py, pz):")
    print(f"       E      px     py     pz")
    labels = ['E', 'px', 'py', 'pz']
    for i, label in enumerate(labels):
        row = '  '.join([f'{corr[i,j]:+.3f}' for j in range(4)])
        print(f"{label:3s}  {row}")
    
    # Check total momentum conservation (center of mass frame)
    total_p = X.sum(axis=1)  # (n_events, 4)
    print(f"\nTotal momentum (should be near zero for CoM frame):")
    print(f"  Mean total E:  {total_p[:, 0].mean():.4e}")
    print(f"  Mean total px: {total_p[:, 1].mean():.4e}")
    print(f"  Mean total py: {total_p[:, 2].mean():.4e}")
    print(f"  Mean total pz: {total_p[:, 3].mean():.4e}")
    
    return corr


def test_input_scale_effect():
    """Check the effect of input scaling."""
    print("\n" + "="*60)
    print("TEST: Input scale effect")
    print("="*60)
    
    input_scale = 0.9515689
    n_events = 100
    n_particles = 128
    
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    X_scaled = X / input_scale
    
    print(f"Original X statistics:")
    print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    print(f"  |X| mean: {np.abs(X).mean():.4f}")
    
    print(f"\nScaled X statistics (scale={input_scale}):")
    print(f"  Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    print(f"  |X| mean: {np.abs(X_scaled).mean():.4f}")
    
    # Check if scaling affects mass
    m2_orig = X[..., 0]**2 - (X[..., 1:]**2).sum(axis=-1)
    m2_scaled = X_scaled[..., 0]**2 - (X_scaled[..., 1:]**2).sum(axis=-1)
    
    print(f"\nMass² before scaling: mean={m2_orig.mean():.4e}")
    print(f"Mass² after scaling:  mean={m2_scaled.mean():.4e}")
    print(f"Ratio: {m2_scaled.mean() / (m2_orig.mean() + 1e-10):.4f}")


def main():
    print("="*80)
    print("DATA PROPERTY ANALYSIS")
    print("="*80)
    print("\nThis analysis checks properties of the training data that might")
    print("explain why layer 1 symmetry doesn't hurt (or helps) performance.")
    
    m2 = test_particle_masses()
    Y, Y_log = test_kp_target_distribution()
    test_what_linear_layer_can_compute()
    test_data_correlation()
    test_input_scale_effect()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. If particles are MASSLESS (m²=0), there is NO non-trivial per-particle 
   Lorentz invariant that a linear layer can compute!
   
2. The only invariants come from PAIRS of particles: s_ij = (p_i + p_j)²

3. If layer 1 is forced to be invariant, it MUST output near-constants
   (just biases, since W → 0)

4. But the model can still work because:
   - Sum pooling aggregates the biases: Σ_i (b) = N * b
   - The rho network then learns to compute the KPs from this "count"
   - Plus residual non-invariant information that leaks through
   
5. The question is: with N*b as input, can rho learn the KPs?
   This seems impossible unless there's something special about the data.
""")


if __name__ == '__main__':
    main()

