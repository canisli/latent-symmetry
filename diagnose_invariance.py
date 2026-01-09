#!/usr/bin/env python3
"""
Diagnostic script to verify layer 1 invariance on trained models.

This script:
1. Loads trained models (baseline, layer1 symmetry, strong layer1 symmetry)
2. Generates test data
3. Applies random Lorentz transforms
4. Measures variance of layer 1 activations
5. Compares task predictions across transforms
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from models import DeepSets
from symmetry import rand_lorentz, lorentz_orbit_variance_loss
from train import load_efp_preset
from data.kp_dataset import make_kp_dataloader

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


def measure_invariance_variance(model, x: torch.Tensor, layer_idx: int, 
                                 n_transforms: int = 10, std_eta: float = 0.5) -> dict:
    """
    Measure the variance of activations under multiple Lorentz transformations.
    
    Returns:
        Dictionary with statistics about activation variance
    """
    B, N, D = x.shape
    
    all_activations = []
    
    with torch.no_grad():
        # Get activations for n_transforms different Lorentz transformations
        for i in range(n_transforms):
            L = rand_lorentz(
                shape=torch.Size([B]),
                std_eta=std_eta,
                n_max_std_eta=3.0,
                device=device,
                dtype=x.dtype,
            )
            
            # Apply Lorentz transformation
            x_rot = torch.matmul(L.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
            
            # Get layer activations
            h = model.forward_with_intermediate(x_rot, layer_idx)
            all_activations.append(h)
    
    # Stack all activations: (n_transforms, B, N, hidden) or (n_transforms, B, hidden)
    stacked = torch.stack(all_activations, dim=0)
    
    # Compute variance across transforms
    variance_per_sample = stacked.var(dim=0)  # Variance across transforms
    mean_variance = variance_per_sample.mean().item()
    max_variance = variance_per_sample.max().item()
    
    # Also measure pairwise differences
    pairwise_diffs = []
    for i in range(n_transforms):
        for j in range(i+1, n_transforms):
            diff = (stacked[i] - stacked[j]).pow(2).sum(dim=-1).mean().item()
            pairwise_diffs.append(diff)
    
    return {
        'mean_variance': mean_variance,
        'max_variance': max_variance,
        'mean_pairwise_mse': np.mean(pairwise_diffs),
        'max_pairwise_mse': np.max(pairwise_diffs),
    }


def measure_output_invariance(model, x: torch.Tensor, 
                               n_transforms: int = 10, std_eta: float = 0.5) -> dict:
    """
    Measure the variance of final outputs under multiple Lorentz transformations.
    
    For Lorentz-invariant targets (kinematic polynomials), the output should be invariant.
    """
    B, N, D = x.shape
    
    all_outputs = []
    
    with torch.no_grad():
        for i in range(n_transforms):
            L = rand_lorentz(
                shape=torch.Size([B]),
                std_eta=std_eta,
                n_max_std_eta=3.0,
                device=device,
                dtype=x.dtype,
            )
            
            x_rot = torch.matmul(L.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
            pred = model(x_rot)
            all_outputs.append(pred)
    
    stacked = torch.stack(all_outputs, dim=0)
    
    variance_per_sample = stacked.var(dim=0)
    mean_variance = variance_per_sample.mean().item()
    
    # Relative variance (normalized by mean output magnitude)
    mean_output = stacked.mean(dim=0)
    relative_variance = (variance_per_sample / (mean_output.abs() + 1e-8)).mean().item()
    
    return {
        'mean_variance': mean_variance,
        'max_variance': variance_per_sample.max().item(),
        'relative_variance': relative_variance,
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose layer invariance in trained models')
    parser.add_argument('--n-events', type=int, default=1000, help='Number of test events')
    parser.add_argument('--n-particles', type=int, default=128, help='Particles per event')
    parser.add_argument('--n-transforms', type=int, default=20, help='Number of Lorentz transforms to sample')
    parser.add_argument('--std-eta', type=float, default=0.5, help='Rapidity std for boosts')
    parser.add_argument('--num-phi-layers', type=int, default=4, help='Number of phi layers')
    parser.add_argument('--num-rho-layers', type=int, default=4, help='Number of rho layers')
    args = parser.parse_args()
    
    # Load EFP preset
    config_dir = Path(__file__).parent / "config"
    edges_list = load_efp_preset('deg3', str(config_dir))
    num_kps = len(edges_list)
    
    # Generate test data
    print(f"Generating {args.n_events} test events with {args.n_particles} particles each...")
    loader = make_kp_dataloader(
        edges_list=edges_list,
        n_events=args.n_events,
        n_particles=args.n_particles,
        batch_size=min(256, args.n_events),
    )
    
    # Get a batch of test data
    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch.to(device)
    print(f"Test batch shape: {x_batch.shape}")
    
    # Define models to analyze
    model_configs = [
        ('4x4_none.pt', 'Baseline (no symmetry)'),
        ('4x4_layer1.pt', 'Layer 1 symmetry'),
        ('4x4_layer1_strong.pt', 'Layer 1 strong symmetry'),
    ]
    
    print(f"\n{'='*80}")
    print("LAYER 1 ACTIVATION INVARIANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Sampling {args.n_transforms} random Lorentz transforms with std_eta={args.std_eta}")
    print(f"{'='*80}\n")
    
    for model_path, model_name in model_configs:
        if not Path(model_path).exists():
            print(f"Skipping {model_name}: {model_path} not found")
            continue
            
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        model = load_model(
            model_path, 
            num_phi_layers=args.num_phi_layers,
            num_rho_layers=args.num_rho_layers,
            num_kps=num_kps,
        )
        
        # Measure invariance at each phi layer
        for layer_idx in range(1, args.num_phi_layers + 1):
            stats = measure_invariance_variance(
                model, x_batch, layer_idx, 
                n_transforms=args.n_transforms, 
                std_eta=args.std_eta
            )
            print(f"\nLayer {layer_idx} (phi):")
            print(f"  Mean variance across transforms: {stats['mean_variance']:.4e}")
            print(f"  Max variance across transforms:  {stats['max_variance']:.4e}")
            print(f"  Mean pairwise MSE:               {stats['mean_pairwise_mse']:.4e}")
        
        # Measure output invariance
        output_stats = measure_output_invariance(
            model, x_batch,
            n_transforms=args.n_transforms,
            std_eta=args.std_eta
        )
        print(f"\nFinal Output:")
        print(f"  Mean variance across transforms:     {output_stats['mean_variance']:.4e}")
        print(f"  Max variance across transforms:      {output_stats['max_variance']:.4e}")
        print(f"  Relative variance (normalized):      {output_stats['relative_variance']:.4e}")
        
        # Compute symmetry loss using the official function
        with torch.no_grad():
            sym_loss_layer1 = lorentz_orbit_variance_loss(
                model, x_batch, layer_idx=1, std_eta=args.std_eta
            ).item()
            sym_loss_output = lorentz_orbit_variance_loss(
                model, x_batch, layer_idx=-1, std_eta=args.std_eta
            ).item()
        
        print(f"\nOfficial Symmetry Loss:")
        print(f"  Layer 1:       {sym_loss_layer1:.4e}")
        print(f"  Output (-1):   {sym_loss_output:.4e}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    print("\nInterpretation:")
    print("- Lower variance = more Lorentz invariant")
    print("- Layer 1 with symmetry training should show lower variance than baseline")
    print("- If output variance is low for all models, they've learned Lorentz-invariant functions")
    print("- High layer 1 variance with low output variance = model compensates in later layers")


if __name__ == '__main__':
    main()

