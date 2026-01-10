"""
Analyze how symmetry emerges across layers in a baseline model (no symmetry penalty).

Computes relative symmetry loss at each hidden layer to investigate whether
symmetry naturally increases with depth, peaking at the output.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from models import MLP
from symmetry import so3_relative_symmetry_loss
from data import ScalarFieldDataset, compute_scalar_field
from train import derive_seed, set_model_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_with_checkpoints(model, loss_fn, train_loader, val_loader, optimizer, 
                           num_epochs=100, checkpoint_fractions=[0, 1/3, 2/3, 1]):
    """
    Train a baseline model and save checkpoints at specified fractions of training.
    
    Args:
        model: Model to train
        loss_fn: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        num_epochs: Total number of epochs
        checkpoint_fractions: List of fractions (0 to 1) at which to save checkpoints
    
    Returns:
        List of (epoch, state_dict) tuples for each checkpoint
    """
    import copy
    
    # Calculate checkpoint epochs
    checkpoint_epochs = [int(f * num_epochs) for f in checkpoint_fractions]
    # Ensure epoch 0 is included for "beginning" and last epoch for "end"
    checkpoint_epochs[0] = 0
    checkpoint_epochs[-1] = num_epochs
    
    checkpoints = []
    
    # Save initial state (epoch 0, before any training)
    if 0 in checkpoint_epochs:
        checkpoints.append((0, copy.deepcopy(model.state_dict())))
        print(f"Saved checkpoint at epoch 0 (before training)")
    
    pbar = tqdm(range(num_epochs), desc="Training baseline")
    
    for epoch in pbar:
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
        val_loss /= len(val_loader)
        
        pbar.set_postfix({'train': f'{train_loss:.2e}', 'val': f'{val_loss:.2e}'})
        
        # Save checkpoint if this is a checkpoint epoch (after training)
        actual_epoch = epoch + 1  # epoch is 0-indexed, but we want 1-indexed for checkpoints
        if actual_epoch in checkpoint_epochs and actual_epoch != 0:
            checkpoints.append((actual_epoch, copy.deepcopy(model.state_dict())))
            print(f"Saved checkpoint at epoch {actual_epoch}")
    
    return checkpoints


def compute_layerwise_symmetry(model, data_loader, num_hidden_layers, num_samples=10, show_progress=True):
    """
    Compute relative symmetry loss at each layer.
    
    Args:
        model: Trained MLP model
        data_loader: DataLoader for evaluation data
        num_hidden_layers: Number of hidden layers in the model
        num_samples: Number of random rotation pairs to average over per batch
        show_progress: Whether to show progress bar
    
    Returns:
        dict mapping layer indices to relative symmetry losses
    """
    model.eval()
    
    # Layer indices: 1 to num_hidden_layers for hidden layers, -1 for output
    layer_indices = list(range(1, num_hidden_layers + 1)) + [-1]
    layer_losses = {idx: [] for idx in layer_indices}
    
    loader = tqdm(data_loader, desc="Computing layerwise symmetry") if show_progress else data_loader
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            
            for layer_idx in layer_indices:
                # Average over multiple random rotation samples for stability
                batch_losses = []
                for _ in range(num_samples):
                    loss = so3_relative_symmetry_loss(model, xb, layer_idx)
                    batch_losses.append(loss.item())
                layer_losses[layer_idx].append(np.mean(batch_losses))
    
    # Average across all batches
    avg_losses = {idx: np.mean(losses) for idx, losses in layer_losses.items()}
    std_losses = {idx: np.std(losses) for idx, losses in layer_losses.items()}
    
    return avg_losses, std_losses


def plot_layerwise_symmetry_evolution(checkpoint_results, num_hidden_layers, num_epochs, save_path=None):
    """
    Create visualization of relative symmetry loss vs layer depth at multiple training stages.
    
    Args:
        checkpoint_results: List of (epoch, avg_losses, std_losses) tuples
        num_hidden_layers: Number of hidden layers
        num_epochs: Total number of training epochs
        save_path: Optional path to save the figure
    """
    # Prepare data for plotting
    layer_indices = list(range(1, num_hidden_layers + 1)) + [-1]
    x_positions = list(range(len(layer_indices)))
    x_labels = [str(i) for i in range(1, num_hidden_layers + 1)] + ['out']
    
    # Color scheme - from light to dark as training progresses
    colors = ['#94a3b8', '#64748b', '#475569', '#1e293b']  # slate palette
    markers = ['o', 's', '^', 'D']
    
    # Create figure with distinctive styling
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    
    # Plot each checkpoint
    for i, (epoch, avg_losses, std_losses) in enumerate(checkpoint_results):
        losses = [avg_losses[idx] for idx in layer_indices]
        stds = [std_losses[idx] for idx in layer_indices]
        
        # Create label
        if epoch == 0:
            label = 'Epoch 0 (untrained)'
        elif epoch == num_epochs:
            label = f'Epoch {epoch} (final)'
        else:
            fraction = epoch / num_epochs
            label = f'Epoch {epoch} ({fraction:.0%} trained)'
        
        # Plot with error bars
        ax.errorbar(x_positions, losses, yerr=stds, 
                    fmt=f'{markers[i]}-', capsize=4, capthick=1.5, 
                    color=colors[i], linewidth=2, markersize=9,
                    markerfacecolor=colors[i], markeredgecolor='white',
                    markeredgewidth=1.5, ecolor=colors[i], alpha=0.9,
                    label=label)
    
    # Styling
    ax.set_xlabel('Layer', fontsize=14, fontweight='medium')
    ax.set_ylabel('Relative Symmetry Loss', fontsize=14, fontweight='medium')
    ax.set_title('SO(3) Symmetry Emergence During Training\n(Baseline: No Symmetry Penalty)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    
    # Grid
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    
    # Tight layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def plot_field_slices(field_type='inv', save_path='slices_inv.png', n_points=100, extent=5.0):
    """
    Plot xy, yz, xz slices of the scalar field.
    
    Args:
        field_type: 'inv' for SO(3) invariant field, 'sph' for non-invariant spherical harmonic
        save_path: Path to save the figure
        n_points: Number of points per dimension for the grid
        extent: Extent of the grid in each direction (from -extent to +extent)
    """
    # Create coordinate grids
    coords = torch.linspace(-extent, extent, n_points)
    
    # XY slice (z=0)
    xx_xy, yy_xy = torch.meshgrid(coords, coords, indexing='ij')
    zz_xy = torch.zeros_like(xx_xy)
    X_xy = torch.stack([xx_xy, yy_xy, zz_xy], dim=-1).reshape(-1, 3)
    field_xy = compute_scalar_field(X_xy, field_type=field_type).reshape(n_points, n_points)
    
    # YZ slice (x=0)
    yy_yz, zz_yz = torch.meshgrid(coords, coords, indexing='ij')
    xx_yz = torch.zeros_like(yy_yz)
    X_yz = torch.stack([xx_yz, yy_yz, zz_yz], dim=-1).reshape(-1, 3)
    field_yz = compute_scalar_field(X_yz, field_type=field_type).reshape(n_points, n_points)
    
    # XZ slice (y=0)
    xx_xz, zz_xz = torch.meshgrid(coords, coords, indexing='ij')
    yy_xz = torch.zeros_like(xx_xz)
    X_xz = torch.stack([xx_xz, yy_xz, zz_xz], dim=-1).reshape(-1, 3)
    field_xz = compute_scalar_field(X_xz, field_type=field_type).reshape(n_points, n_points)
    
    # Create 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find common color scale for all plots
    vmin = min(field_xy.min().item(), field_yz.min().item(), field_xz.min().item())
    vmax = max(field_xy.max().item(), field_yz.max().item(), field_xz.max().item())
    
    # Plot XY slice
    im1 = axes[0].imshow(field_xy.numpy(), extent=[-extent, extent, -extent, extent], 
                        origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('XY Slice (z=0)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('y', fontsize=11)
    plt.colorbar(im1, ax=axes[0])
    
    # Plot YZ slice
    im2 = axes[1].imshow(field_yz.numpy(), extent=[-extent, extent, -extent, extent], 
                        origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('YZ Slice (x=0)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('y', fontsize=11)
    axes[1].set_ylabel('z', fontsize=11)
    plt.colorbar(im2, ax=axes[1])
    
    # Plot XZ slice
    im3 = axes[2].imshow(field_xz.numpy(), extent=[-extent, extent, -extent, extent], 
                        origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title('XZ Slice (y=0)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('z', fontsize=11)
    plt.colorbar(im3, ax=axes[2])
    
    # Add overall title
    field_type_label = 'SO(3) Invariant' if field_type == 'inv' else 'Spherical Harmonic'
    fig.suptitle(f'Scalar Field Slices: {field_type_label}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved field slices to {save_path}")
    plt.close()


def main(num_hidden_layers=6, hidden_dim=128, learning_rate=3e-4, 
         num_epochs=100, run_seed=42, save_path='layerwise_symmetry.png', field_type='inv'):
    """
    Main function to train baseline and analyze layerwise symmetry at multiple checkpoints.
    
    Args:
        num_hidden_layers: Number of hidden layers in MLP
        hidden_dim: Dimension of each hidden layer
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        run_seed: Random seed for reproducibility
        save_path: Path to save the visualization
        field_type: 'inv' for SO(3) invariant field, 'sph' for non-invariant spherical harmonic
    """
    print(f"Device: {device}")
    print(f"Architecture: {num_hidden_layers} hidden layers × {hidden_dim} units")
    print(f"Field type: {field_type}")
    print(f"Training for {num_epochs} epochs with lr={learning_rate}")
    print()
    
    # Modify save_path to include field_type
    base, ext = os.path.splitext(save_path)
    # If it's the default path, replace it; otherwise insert field_type before extension
    if base == 'layerwise_symmetry':
        save_path = f'layerwise_symmetry_{field_type}{ext}'
    else:
        save_path = f'{base}_{field_type}{ext}'
    
    # Derive seeds
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    
    # Create dataset
    field = ScalarFieldDataset(100000, seed=data_seed, field_type=field_type)
    batch_size = 128
    
    generator = torch.Generator().manual_seed(data_seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        field, [0.6, 0.2, 0.2], generator=generator
    )
    
    loader_generator = torch.Generator().manual_seed(data_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Create model
    set_model_seed(model_seed)
    dims = [3] + [hidden_dim] * num_hidden_layers + [1]
    model = MLP(dims).to(device)
    
    print(f"Model architecture: {dims}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train with checkpoints at 0%, 33%, 67%, 100%
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    checkpoints = train_with_checkpoints(
        model, loss_fn, train_loader, val_loader, optimizer, 
        num_epochs=num_epochs, checkpoint_fractions=[0, 1/3, 2/3, 1]
    )
    
    # Compute layerwise symmetry at each checkpoint
    print("\nAnalyzing symmetry at each checkpoint...")
    checkpoint_results = []
    
    for epoch, state_dict in checkpoints:
        print(f"\n  Evaluating checkpoint at epoch {epoch}...")
        model.load_state_dict(state_dict)
        avg_losses, std_losses = compute_layerwise_symmetry(
            model, test_loader, num_hidden_layers, num_samples=10, show_progress=False
        )
        checkpoint_results.append((epoch, avg_losses, std_losses))
        
        # Print results for this checkpoint
        print(f"    Layer 1: {avg_losses[1]:.4f}, Output: {avg_losses[-1]:.4f}")
    
    # Print final summary
    print("\n" + "="*60)
    print("LAYERWISE RELATIVE SYMMETRY LOSS EVOLUTION")
    print("="*60)
    layer_indices = list(range(1, num_hidden_layers + 1)) + [-1]
    
    # Header
    header = "Epoch".ljust(12) + "".join([f"L{i}".rjust(10) for i in range(1, num_hidden_layers + 1)]) + "Output".rjust(10)
    print(header)
    print("-" * len(header))
    
    for epoch, avg_losses, _ in checkpoint_results:
        row = f"{epoch}".ljust(12)
        for idx in layer_indices:
            row += f"{avg_losses[idx]:.4f}".rjust(10)
        print(row)
    print("="*60)
    
    # Generate visualization
    fig, ax = plot_layerwise_symmetry_evolution(checkpoint_results, num_hidden_layers, num_epochs, save_path)
    plt.show()
    
    return checkpoint_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze layerwise symmetry emergence in baseline model')
    parser.add_argument('--num-hidden-layers', type=int, default=6,
                        help='Number of hidden layers (default: 6)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Size of each hidden layer (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for optimizer (default: 3e-4)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--run-seed', type=int, default=42,
                        help='Base random seed for reproducibility (default: 42)')
    parser.add_argument('--save-path', type=str, default='layerwise_symmetry.png',
                        help='Path to save the visualization (default: layerwise_symmetry.png)')
    parser.add_argument('--field-type', type=str, default='inv', choices=['inv', 'sph'],
                        help='Field type: "inv" for SO(3) invariant field, "sph" for non-invariant spherical harmonic (default: inv)')
    
    args = parser.parse_args()
    
    main(
        num_hidden_layers=args.num_hidden_layers,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        run_seed=args.run_seed,
        save_path=args.save_path,
        field_type=args.field_type
    )
