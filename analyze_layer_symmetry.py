#!/usr/bin/env python3
"""
Diagnose how Lorentz symmetry emerges through network depth.

For a no-penalty baseline model, evaluates relative symmetry loss at each hidden
layer to test whether symmetry naturally increases with depth.

The relative symmetry loss is:
    ||a - b||^2 / (||a||^2 + ||b||^2 + eps)

This normalization makes losses comparable across layers with different activation scales.
Values are bounded in [0, 2]: 0 = perfect invariance, 2 = opposite directions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from models import DeepSets
from data.kp_dataset import make_kp_dataloader
from symmetry import rand_lorentz
from train import load_efp_preset


# ============================================================================
# Relative Symmetry Loss Function
# ============================================================================

def relative_symmetry_loss(
    model,
    x: torch.Tensor,
    layer_idx: int,
    std_eta: float = 0.5,
    n_max_std_eta: float = 3.0,
    generator: torch.Generator = None,
    mask: torch.Tensor = None,
    eps: float = 1e-8,
):
    """
    Compute RELATIVE symmetry loss at a given layer.
    
    The relative loss is: ||a - b||^2 / (||a||^2 + ||b||^2 + eps)
    
    This makes losses comparable across layers since it normalizes by activation scale.
    Values are bounded in [0, 2]: 0 = perfect invariance.
    
    Args:
        model: Model with forward_with_intermediate() method
        x: Input 4-vectors of shape (batch_size, num_particles, 4)
        layer_idx: Which layer's activations to compare (-1 for output)
        std_eta: Standard deviation of rapidity for boosts
        n_max_std_eta: Maximum number of standard deviations for truncation
        generator: Optional random generator for reproducibility
        mask: Optional boolean mask of shape (batch_size, num_particles)
        eps: Small constant for numerical stability
    
    Returns:
        Scalar relative symmetry loss
    """
    B, N, D = x.shape
    assert D == 4, f"Expected 4-vectors, got dimension {D}"
    device, dtype = x.device, x.dtype
    
    # Sample two random Lorentz transformations
    L1 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    L2 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    
    # Apply Lorentz transformations to all particles
    x_rot1 = torch.matmul(L1.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.matmul(L2.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    # Get intermediate activations
    h1 = model.forward_with_intermediate(x_rot1, layer_idx, mask=mask)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx, mask=mask)
    
    # Handle per-particle representations (pre-pooling layers)
    if h1.dim() == 3:  # (B, N, hidden)
        if mask is None:
            mask = torch.any(x != 0.0, dim=-1)  # (B, N)
        
        # Compute squared norms and differences per particle
        # h1, h2: (B, N, hidden)
        diff = h1 - h2
        diff_sq = diff.pow(2).sum(dim=-1)  # (B, N)
        h1_sq = h1.pow(2).sum(dim=-1)  # (B, N)
        h2_sq = h2.pow(2).sum(dim=-1)  # (B, N)
        
        # Relative loss per particle: ||a-b||^2 / (||a||^2 + ||b||^2 + eps)
        per_particle_rel_loss = diff_sq / (h1_sq + h2_sq + eps)  # (B, N)
        
        # Average over valid particles
        mask_float = mask.float()
        total_valid = mask_float.sum().clamp(min=1.0)
        loss = (per_particle_rel_loss * mask_float).sum() / total_valid
        
        return loss
    
    # Post-pooling layers: h1 and h2 have shape (B, hidden)
    diff = h1 - h2
    diff_sq = diff.pow(2).sum(dim=-1)  # (B,)
    h1_sq = h1.pow(2).sum(dim=-1)  # (B,)
    h2_sq = h2.pow(2).sum(dim=-1)  # (B,)
    
    # Relative loss per sample
    per_sample_rel_loss = diff_sq / (h1_sq + h2_sq + eps)  # (B,)
    loss = per_sample_rel_loss.mean()
    
    return loss


# ============================================================================
# Diagnostic Functions
# ============================================================================

def evaluate_layer_symmetry(
    model,
    dataloader,
    layer_idx: int,
    std_eta: float = 0.5,
    n_samples: int = 5,
    device: str = 'cuda',
):
    """
    Evaluate relative symmetry loss at a specific layer over the dataset.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        layer_idx: Layer index to evaluate
        std_eta: Rapidity std for Lorentz transforms
        n_samples: Number of random transform pairs to sample per batch
        device: Device to use
    
    Returns:
        mean_loss: Mean relative symmetry loss
        std_loss: Standard deviation across batches
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            
            # Sample multiple transform pairs for more robust estimate
            batch_losses = []
            for _ in range(n_samples):
                loss = relative_symmetry_loss(
                    model, xb, layer_idx, std_eta=std_eta
                )
                batch_losses.append(loss.item())
            
            losses.append(np.mean(batch_losses))
    
    return np.mean(losses), np.std(losses) / np.sqrt(len(losses))


def get_layer_names(num_phi_layers: int, num_rho_layers: int):
    """Generate descriptive names for each layer index."""
    names = {}
    
    # Phi layers (per-particle)
    for i in range(1, num_phi_layers + 1):
        names[i] = f"phi_{i}"
    
    # Pooling layer
    pool_idx = num_phi_layers + 1
    names[pool_idx] = "pool"
    
    # Rho layers (post-pooling)
    for i in range(1, num_rho_layers + 1):
        rho_idx = pool_idx + i
        names[rho_idx] = f"rho_{i}"
    
    # Output
    names[-1] = "output"
    
    return names


def diagnose_model(
    model_path: str,
    num_events: int = 2000,
    n_particles: int = 128,
    batch_size: int = 256,
    input_scale: float = 0.9515689,
    std_eta: float = 0.5,
    n_samples: int = 5,
    device: str = None,
    target_type: str = 'kinematic',
    efp_beta: float = 2.0,
    efp_kappa: float = 1.0,
    efp_normed: bool = False,
    target_transform: str = 'log1p',
):
    """
    Run full layer-wise symmetry diagnosis on a trained model.
    
    Args:
        model_path: Path to saved model weights
        num_events: Number of events to evaluate on
        n_particles: Particles per event
        batch_size: Batch size for evaluation
        input_scale: Data input scale
        std_eta: Rapidity std for Lorentz transforms
        n_samples: Transform samples per batch
        device: Device to use (auto-detect if None)
        target_type: 'kinematic' (Lorentz invariant) or 'efp' (non-invariant)
        efp_beta: Angular weighting for EFP measure (only used if target_type='efp')
        efp_kappa: Energy weighting for EFP measure (only used if target_type='efp')
        efp_normed: Whether to normalize energies for EFP (only used if target_type='efp')
        target_transform: How targets are transformed ('log1p', 'log_standardized', 'standardized')
    
    Returns:
        results: Dict mapping layer_idx -> (mean_loss, std_err, layer_name)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model architecture (matching 4x4 models)
    num_phi_layers = 4
    num_rho_layers = 4
    hidden_channels = 128
    
    # Load EFP preset to determine output channels
    edges_list = load_efp_preset('deg3', 'config')
    num_kps = len(edges_list)
    
    # Create model
    model = DeepSets(
        in_channels=4,
        out_channels=num_kps,
        hidden_channels=hidden_channels,
        num_phi_layers=num_phi_layers,
        num_rho_layers=num_rho_layers,
        pool_mode='sum',
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Determine measure parameters based on target type
    if target_type == 'kinematic':
        measure = 'kinematic'
        beta = 2.0  # ignored by kinematic measure
        kappa = 1.0  # ignored by kinematic measure
        normed = False  # ignored by kinematic measure
        print(f"Target type: kinematic (Lorentz invariant)")
    elif target_type == 'efp':
        measure = 'eeefm'
        beta = efp_beta
        kappa = efp_kappa
        normed = efp_normed
        normed_str = "normed" if normed else "unnormed"
        print(f"Target type: EFP (eeefm, beta={beta}, kappa={kappa}, {normed_str}) - NOT Lorentz invariant")
    else:
        raise ValueError(f"Unknown target_type: {target_type}. Must be 'kinematic' or 'efp'")
    
    # Create test dataloader with appropriate measure
    dataloader = make_kp_dataloader(
        edges_list=edges_list,
        n_events=num_events,
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        measure=measure,
        beta=beta,
        kappa=kappa,
        normed=normed,
        target_transform=target_transform,
    )
    
    # Get layer names
    layer_names = get_layer_names(num_phi_layers, num_rho_layers)
    
    # Evaluate each layer
    results = {}
    max_layer_idx = num_phi_layers + num_rho_layers + 1
    
    # All layer indices including output
    layer_indices = list(range(1, max_layer_idx + 1)) + [-1]
    
    print(f"\nEvaluating relative symmetry at {len(layer_indices)} layers...")
    print(f"  std_eta = {std_eta}, n_samples = {n_samples}")
    print(f"  {num_events} events, {n_particles} particles/event\n")
    
    for layer_idx in tqdm(layer_indices, desc="Layers"):
        mean_loss, std_err = evaluate_layer_symmetry(
            model, dataloader, layer_idx,
            std_eta=std_eta, n_samples=n_samples, device=device,
        )
        name = layer_names.get(layer_idx, f"layer_{layer_idx}")
        results[layer_idx] = (mean_loss, std_err, name)
    
    return results


def print_results(results: dict):
    """Print results table."""
    print("\n" + "=" * 60)
    print("LAYER-WISE RELATIVE SYMMETRY LOSS")
    print("=" * 60)
    print(f"{'Layer':<12} {'Index':<8} {'Rel. Sym Loss':<18} {'Std Err':<12}")
    print("-" * 60)
    
    # Sort by layer index (but put -1 at end)
    sorted_keys = sorted([k for k in results.keys() if k > 0]) + [-1]
    
    for layer_idx in sorted_keys:
        mean_loss, std_err, name = results[layer_idx]
        idx_str = str(layer_idx) if layer_idx > 0 else "output"
        print(f"{name:<12} {idx_str:<8} {mean_loss:<18.6f} {std_err:<12.6f}")
    
    print("=" * 60)
    
    # Analysis
    phi_losses = [results[i][0] for i in sorted_keys if results[i][2].startswith('phi')]
    rho_losses = [results[i][0] for i in sorted_keys if results[i][2].startswith('rho')]
    pool_loss = results[sorted_keys[len(phi_losses)]][0]  # Pool layer
    output_loss = results[-1][0]
    
    print("\nSUMMARY:")
    print(f"  Phi layers (per-particle): {np.mean(phi_losses):.6f} avg")
    print(f"  Pool layer:                {pool_loss:.6f}")
    print(f"  Rho layers (post-pool):    {np.mean(rho_losses):.6f} avg")
    print(f"  Output:                    {output_loss:.6f}")
    
    # Trend analysis
    all_losses = phi_losses + [pool_loss] + rho_losses + [output_loss]
    if all_losses[0] > all_losses[-1]:
        print("\n  → Symmetry INCREASES with depth (loss decreases)")
    else:
        print("\n  → Symmetry DECREASES with depth (loss increases)")


def plot_results(results: dict, save_path: str = None):
    """Plot relative symmetry vs layer depth."""
    
    # Sort by layer index
    sorted_keys = sorted([k for k in results.keys() if k > 0]) + [-1]
    
    names = [results[k][2] for k in sorted_keys]
    means = [results[k][0] for k in sorted_keys]
    stds = [results[k][1] for k in sorted_keys]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, 
                  color=['#3498db'] * 4 + ['#2ecc71'] + ['#e74c3c'] * 4 + ['#9b59b6'],
                  edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Relative Symmetry Loss', fontsize=12)
    ax.set_title('Lorentz Symmetry Emergence Through Network Depth\n(Lower = More Invariant)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add horizontal lines for reference
    # ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect invariance')
    # ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Maximum (opposite)')
    
    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Phi (per-particle)'),
        Patch(facecolor='#2ecc71', label='Pool'),
        Patch(facecolor='#e74c3c', label='Rho (post-pool)'),
        Patch(facecolor='#9b59b6', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose layer-wise symmetry emergence')
    parser.add_argument('--model', type=str, default='4x4_none.pt',
                        help='Path to trained model weights')
    parser.add_argument('--num-events', type=int, default=2000,
                        help='Number of events for evaluation')
    parser.add_argument('--std-eta', type=float, default=0.5,
                        help='Rapidity std for Lorentz transforms')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of transform samples per batch')
    parser.add_argument('--save-plot', type=str, default='layer_symmetry.png',
                        help='Path to save plot')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    # Target type arguments
    parser.add_argument('--target-type', type=str, default='kinematic',
                        choices=['kinematic', 'efp'],
                        help='Target type: kinematic (Lorentz invariant) or efp (non-invariant)')
    parser.add_argument('--efp-beta', type=float, default=2.0,
                        help='EFP angular weighting exponent (only used with --target-type efp)')
    parser.add_argument('--efp-kappa', type=float, default=1.0,
                        help='EFP energy weighting exponent (only used with --target-type efp)')
    parser.add_argument('--efp-normed', action='store_true',
                        help='Normalize energies for EFP (default: False)')
    parser.add_argument('--target-transform', type=str, default='log1p',
                        choices=['log1p', 'log_standardized', 'standardized'],
                        help='Target transformation (must match training)')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("\nAvailable models:")
        for p in Path('.').glob('*.pt'):
            print(f"  {p}")
        return
    
    # Run diagnosis
    results = diagnose_model(
        model_path=args.model,
        num_events=args.num_events,
        std_eta=args.std_eta,
        n_samples=args.n_samples,
        target_type=args.target_type,
        efp_beta=args.efp_beta,
        efp_kappa=args.efp_kappa,
        efp_normed=args.efp_normed,
        target_transform=args.target_transform,
    )
    
    # Print results
    print_results(results)
    
    # Plot results
    if not args.no_plot:
        plot_results(results, save_path=args.save_plot)


if __name__ == '__main__':
    main()

