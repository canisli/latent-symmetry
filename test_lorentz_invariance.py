#!/usr/bin/env python3
"""
Verify Lorentz invariance properties of different EFP measures.

This script tests:
- Kinematic polynomials (measure='kinematic'): Should be Lorentz INVARIANT
- EFPs with eeefm measure (unnormed): Should NOT be Lorentz invariant

The test applies random Lorentz boosts to particle momenta and checks whether
the computed values change.
"""

import numpy as np
import energyflow as ef
from data.kp_dataset import compute_kps
from train import load_efp_preset


def random_lorentz_boost(p4s: np.ndarray, eta_std: float = 0.5) -> np.ndarray:
    """
    Apply a random Lorentz boost to 4-momenta.
    
    Args:
        p4s: Array of shape (N, 4) with columns [E, px, py, pz]
        eta_std: Standard deviation of rapidity for boost
    
    Returns:
        Boosted 4-momenta of same shape
    """
    # Sample random boost direction (unit 3-vector)
    direction = np.random.randn(3)
    direction = direction / np.linalg.norm(direction)
    
    # Sample random rapidity
    eta = np.random.randn() * eta_std
    
    # Compute boost parameters
    gamma = np.cosh(eta)
    beta_gamma = np.sinh(eta)
    beta = np.tanh(eta)
    
    # Build Lorentz boost matrix (in [E, px, py, pz] convention)
    # For boost along direction n with velocity β:
    # Λ^0_0 = γ
    # Λ^0_i = Λ^i_0 = -βγ n_i
    # Λ^i_j = δ_ij + (γ-1) n_i n_j
    
    Lambda = np.eye(4)
    Lambda[0, 0] = gamma
    Lambda[0, 1:] = -beta_gamma * direction
    Lambda[1:, 0] = -beta_gamma * direction
    Lambda[1:, 1:] = np.eye(3) + (gamma - 1) * np.outer(direction, direction)
    
    # Apply boost: p' = Λ p
    return (Lambda @ p4s.T).T


def test_invariance(
    edges_list,
    measure: str,
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
    n_events: int = 100,
    n_particles: int = 32,
    n_boosts: int = 10,
    eta_std: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Test Lorentz invariance of EFP values under random boosts.
    
    Returns dict with:
        - mean_rel_diff: Mean relative difference after boost
        - max_rel_diff: Maximum relative difference
        - is_invariant: True if mean_rel_diff < 1e-5
    """
    np.random.seed(seed)
    
    # Generate random events
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    # Compute original values
    Y_original = compute_kps(X, edges_list, measure=measure, coords='epxpypz', 
                              beta=beta, kappa=kappa, normed=normed)
    
    rel_diffs = []
    
    for _ in range(n_boosts):
        # Apply random boost to each event
        X_boosted = np.zeros_like(X)
        for i in range(n_events):
            # Get non-zero particles
            mask = np.any(X[i] != 0.0, axis=1)
            X_boosted[i, mask] = random_lorentz_boost(X[i, mask], eta_std=eta_std)
        
        # Compute values on boosted momenta
        Y_boosted = compute_kps(X_boosted, edges_list, measure=measure, coords='epxpypz',
                                 beta=beta, kappa=kappa, normed=normed)
        
        # Compute relative difference
        # Use max of original and boosted as denominator to handle near-zero values
        denom = np.maximum(np.abs(Y_original), np.abs(Y_boosted)) + 1e-10
        rel_diff = np.abs(Y_original - Y_boosted) / denom
        rel_diffs.append(rel_diff)
    
    rel_diffs = np.array(rel_diffs)
    mean_rel_diff = np.mean(rel_diffs)
    max_rel_diff = np.max(rel_diffs)
    
    return {
        'mean_rel_diff': mean_rel_diff,
        'max_rel_diff': max_rel_diff,
        'is_invariant': mean_rel_diff < 1e-5,
    }


def main():
    print("=" * 70)
    print("LORENTZ INVARIANCE TEST")
    print("=" * 70)
    print()
    print("Testing whether different EFP measures are Lorentz invariant.")
    print("We apply random Lorentz boosts and measure how much values change.")
    print()
    
    # Load EFP preset
    edges_list = load_efp_preset('deg3', 'config')
    print(f"Using {len(edges_list)} polynomial graphs from 'deg3' preset")
    print()
    
    # Test parameters
    n_events = 100
    n_particles = 32
    n_boosts = 10
    eta_std = 0.5
    
    print(f"Parameters: {n_events} events, {n_particles} particles, {n_boosts} boosts, η_std={eta_std}")
    print()
    
    # Test 1: Kinematic measure (should be invariant)
    print("-" * 70)
    print("TEST 1: KINEMATIC MEASURE")
    print("-" * 70)
    result_kinematic = test_invariance(
        edges_list=edges_list,
        measure='kinematic',
        n_events=n_events,
        n_particles=n_particles,
        n_boosts=n_boosts,
        eta_std=eta_std,
    )
    print(f"  Mean relative difference: {result_kinematic['mean_rel_diff']:.2e}")
    print(f"  Max relative difference:  {result_kinematic['max_rel_diff']:.2e}")
    print(f"  Is Lorentz invariant:     {result_kinematic['is_invariant']}")
    if result_kinematic['is_invariant']:
        print("  ✓ PASS: Kinematic polynomials ARE Lorentz invariant")
    else:
        print("  ✗ FAIL: Kinematic polynomials should be Lorentz invariant!")
    print()
    
    # Test 2: EEEFM measure, unnormed (should NOT be invariant)
    print("-" * 70)
    print("TEST 2: EEEFM MEASURE (unnormed)")
    print("-" * 70)
    result_eeefm_unnormed = test_invariance(
        edges_list=edges_list,
        measure='eeefm',
        beta=2.0,
        kappa=1.0,
        normed=False,
        n_events=n_events,
        n_particles=n_particles,
        n_boosts=n_boosts,
        eta_std=eta_std,
    )
    print(f"  Mean relative difference: {result_eeefm_unnormed['mean_rel_diff']:.2e}")
    print(f"  Max relative difference:  {result_eeefm_unnormed['max_rel_diff']:.2e}")
    print(f"  Is Lorentz invariant:     {result_eeefm_unnormed['is_invariant']}")
    if not result_eeefm_unnormed['is_invariant']:
        print("  ✓ PASS: EEEFM (unnormed) is NOT Lorentz invariant (as expected)")
    else:
        print("  ✗ FAIL: EEEFM should NOT be Lorentz invariant!")
    print()
    
    # Test 3: EEEFM measure, normed (should also NOT be invariant)
    print("-" * 70)
    print("TEST 3: EEEFM MEASURE (normed)")
    print("-" * 70)
    result_eeefm_normed = test_invariance(
        edges_list=edges_list,
        measure='eeefm',
        beta=2.0,
        kappa=1.0,
        normed=True,
        n_events=n_events,
        n_particles=n_particles,
        n_boosts=n_boosts,
        eta_std=eta_std,
    )
    print(f"  Mean relative difference: {result_eeefm_normed['mean_rel_diff']:.2e}")
    print(f"  Max relative difference:  {result_eeefm_normed['max_rel_diff']:.2e}")
    print(f"  Is Lorentz invariant:     {result_eeefm_normed['is_invariant']}")
    if not result_eeefm_normed['is_invariant']:
        print("  ✓ PASS: EEEFM (normed) is NOT Lorentz invariant (as expected)")
    else:
        print("  ? NOTE: EEEFM (normed) appears invariant - may have low variance")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = (
        result_kinematic['is_invariant'] and 
        not result_eeefm_unnormed['is_invariant']
    )
    if all_pass:
        print("✓ All tests passed!")
        print()
        print("  - Kinematic polynomials: Lorentz INVARIANT (good for symmetry learning)")
        print("  - EEEFM (unnormed):      NOT invariant (control experiment)")
    else:
        print("✗ Some tests failed - check results above")
    print()


if __name__ == '__main__':
    main()

