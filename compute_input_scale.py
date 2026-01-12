#!/usr/bin/env python3
"""
Compute and print scale values for hardcoding into config.yaml.

This script estimates:
1. Input scale: global scale for four-momenta so inputs land near unit std
2. Target scales: scale of raw (pre-log1p) target values for different measures

Run this once to get the values, then hardcode them in config.yaml.
"""

from data.kp_dataset import estimate_input_scale, estimate_target_scale
from train import load_efp_preset

# Use same parameters as config defaults
n_events = 2000  # Same as previous input_scale_events
n_particles = 128
seed = 42  # Use a fixed seed for reproducibility

# Load default EFP preset
edges_list = load_efp_preset('deg3', 'config')

print("=" * 60)
print("SCALE COMPUTATION")
print("=" * 60)
print(f"Parameters: n_events={n_events}, n_particles={n_particles}, seed={seed}")
print(f"EFP preset: deg3 ({len(edges_list)} polynomials)")
print("=" * 60)

# Compute input scale (four-momenta scale)
input_scale = estimate_input_scale(n_events=n_events, n_particles=n_particles, seed=seed)
print(f"\n1. Input scale (four-momenta std):")
print(f"   input_scale: {input_scale:.6e}")

# Compute target scale for kinematic measure (Lorentz invariant)
target_scale_kinematic = estimate_target_scale(
    edges_list=edges_list,
    n_events=n_events,
    n_particles=n_particles,
    measure='kinematic',
    seed=seed,
)
print(f"\n2. Target scale for KINEMATIC measure (Lorentz invariant):")
print(f"   target_scale: {target_scale_kinematic:.6e}")

# Compute target scale for eeefm measure (NOT Lorentz invariant, unnormed)
target_scale_eeefm = estimate_target_scale(
    edges_list=edges_list,
    n_events=n_events,
    n_particles=n_particles,
    measure='eeefm',
    beta=2.0,
    kappa=1.0,
    normed=False,  # Use unnormed for varied target values
    seed=seed,
)
print(f"\n3. Target scale for EEEFM measure (NOT Lorentz invariant, unnormed):")
print(f"   target_scale: {target_scale_eeefm:.6e}")

print("\n" + "=" * 60)
print("CONFIG VALUES")
print("=" * 60)
print(f"""
Update config.yaml with:

data:
  input_scale: {input_scale:.6e}  # Four-momenta std (n_events={n_events}, n_particles={n_particles}, seed={seed})
  
# Note: Target scales are informational only.
# Kinematic target scale: {target_scale_kinematic:.6e}
# EEEFM target scale: {target_scale_eeefm:.6e}
# Scale ratio (eeefm/kinematic): {target_scale_eeefm/target_scale_kinematic:.2f}x
""")
