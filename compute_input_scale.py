#!/usr/bin/env python3
"""
Compute and print the input_scale value for hardcoding into config.yaml.

This script estimates the global scale for four-momenta so inputs land near unit std.
Run this once to get the value, then hardcode it in config.yaml.
"""

from data.kp_dataset import estimate_input_scale

# Use same parameters as config defaults
n_events = 2000  # Same as previous input_scale_events
n_particles = 128
seed = 42  # Use a fixed seed for reproducibility

input_scale = estimate_input_scale(n_events=n_events, n_particles=n_particles, seed=seed)

print(f"Computed input_scale: {input_scale:.6e}")
print(f"\nUpdate config.yaml with:")
print(f"  input_scale: {input_scale:.6e}  # Computed with n_events={n_events}, n_particles={n_particles}, seed={seed}")

