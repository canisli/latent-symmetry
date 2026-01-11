"""
Analyze how SO(3) symmetry emerges across layers when training on convex hull volume prediction.

Compares two architectures:
1. DeepSets: Permutation-invariant architecture (phi -> pool -> rho)
2. MLP: Standard MLP with flattened input

Both are trained without symmetry penalty (baseline) to observe natural symmetry emergence.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import copy

from models import MLP, DeepSets
from symmetry_sets import so3_relative_symmetry_loss_sets, so3_relative_symmetry_loss_mlp_sets
from data_convex_hull import ConvexHullDataset, ConvexHullDatasetFlattened
from train import derive_seed, set_model_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_with_checkpoints(model, train_loader, val_loader, optimizer, 
                           num_epochs=100, checkpoint_fractions=[0, 1/3, 2/3, 1],
                           model_type='deepsets'):
    """
    Train a baseline model and save checkpoints at specified fractions of training.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        num_epochs: Total number of epochs
        checkpoint_fractions: List of fractions (0 to 1) at which to save checkpoints
        model_type: 'deepsets' or 'mlp'
    
    Returns:
        checkpoints: List of (epoch, state_dict) tuples for each checkpoint
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    loss_fn = torch.nn.MSELoss()
    
    # Calculate checkpoint epochs
    checkpoint_epochs = [int(f * num_epochs) for f in checkpoint_fractions]
    checkpoint_epochs[0] = 0
    checkpoint_epochs[-1] = num_epochs
    
    checkpoints = []
    train_losses = []
    val_losses = []
    
    # Save initial state (epoch 0, before any training)
    if 0 in checkpoint_epochs:
        checkpoints.append((0, copy.deepcopy(model.state_dict())))
        print(f"Saved checkpoint at epoch 0 (before training)")
    
    pbar = tqdm(range(num_epochs), desc=f"Training {model_type}")
    
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
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        pbar.set_postfix({'train': f'{train_loss:.2e}', 'val': f'{val_loss:.2e}'})
        
        # Save checkpoint if this is a checkpoint epoch (after training)
        actual_epoch = epoch + 1
        if actual_epoch in checkpoint_epochs and actual_epoch != 0:
            checkpoints.append((actual_epoch, copy.deepcopy(model.state_dict())))
            print(f"Saved checkpoint at epoch {actual_epoch}")
    
    return checkpoints, train_losses, val_losses


def compute_layerwise_symmetry_deepsets(model, data_loader, num_samples=10, show_progress=True):
    """
    Compute relative symmetry loss at each layer for DeepSets.
    
    Args:
        model: DeepSets model
        data_loader: DataLoader for evaluation data
        num_samples: Number of random rotation pairs to average over per batch
        show_progress: Whether to show progress bar
    
    Returns:
        avg_losses: dict mapping layer indices to relative symmetry losses
        std_losses: dict mapping layer indices to standard deviations
    """
    model.eval()
    
    # Layer indices: 1 to total_layers, plus -1 for output
    layer_indices = list(range(1, model.total_layers + 1)) + [-1]
    layer_losses = {idx: [] for idx in layer_indices}
    
    loader = tqdm(data_loader, desc="Computing symmetry") if show_progress else data_loader
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            
            for layer_idx in layer_indices:
                batch_losses = []
                for _ in range(num_samples):
                    loss = so3_relative_symmetry_loss_sets(model, xb, layer_idx)
                    batch_losses.append(loss.item())
                layer_losses[layer_idx].append(np.mean(batch_losses))
    
    avg_losses = {idx: np.mean(losses) for idx, losses in layer_losses.items()}
    std_losses = {idx: np.std(losses) for idx, losses in layer_losses.items()}
    
    return avg_losses, std_losses


def compute_layerwise_symmetry_mlp(model, data_loader, n_points, num_samples=10, show_progress=True):
    """
    Compute relative symmetry loss at each layer for MLP.
    
    Args:
        model: MLP model
        data_loader: DataLoader for evaluation data (flattened)
        n_points: Number of points per sample
        num_samples: Number of random rotation pairs to average over per batch
        show_progress: Whether to show progress bar
    
    Returns:
        avg_losses: dict mapping layer indices to relative symmetry losses
        std_losses: dict mapping layer indices to standard deviations
    """
    model.eval()
    
    # Layer indices: 1 to num_linear_layers, plus -1 for output
    layer_indices = list(range(1, model.num_linear_layers + 1)) + [-1]
    layer_losses = {idx: [] for idx in layer_indices}
    
    loader = tqdm(data_loader, desc="Computing symmetry") if show_progress else data_loader
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            
            for layer_idx in layer_indices:
                batch_losses = []
                for _ in range(num_samples):
                    loss = so3_relative_symmetry_loss_mlp_sets(model, xb, n_points, layer_idx)
                    batch_losses.append(loss.item())
                layer_losses[layer_idx].append(np.mean(batch_losses))
    
    avg_losses = {idx: np.mean(losses) for idx, losses in layer_losses.items()}
    std_losses = {idx: np.std(losses) for idx, losses in layer_losses.items()}
    
    return avg_losses, std_losses


def plot_training_curves(deepsets_train, deepsets_val, mlp_train, mlp_val, save_path='convex_hull_training.png'):
    """
    Plot training and validation loss curves for both architectures.
    
    Args:
        deepsets_train: List of training losses per epoch for DeepSets
        deepsets_val: List of validation losses per epoch for DeepSets
        mlp_train: List of training losses per epoch for MLP
        mlp_val: List of validation losses per epoch for MLP
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = np.arange(1, len(deepsets_train) + 1)
    
    # DeepSets
    ax = axes[0]
    ax.set_facecolor('#f8f9fa')
    ax.plot(epochs, deepsets_train, color='#3b82f6', linewidth=2, label='Train')
    ax.plot(epochs, deepsets_val, color='#ef4444', linewidth=2, label='Validation')
    ax.set_xlabel('Epoch', fontsize=13, fontweight='medium')
    ax.set_ylabel('MSE Loss', fontsize=13, fontweight='medium')
    ax.set_title('DeepSets', fontsize=15, fontweight='bold', pad=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.tick_params(axis='both', labelsize=11)
    
    # MLP
    ax = axes[1]
    ax.set_facecolor('#f8f9fa')
    ax.plot(epochs, mlp_train, color='#3b82f6', linewidth=2, label='Train')
    ax.plot(epochs, mlp_val, color='#ef4444', linewidth=2, label='Validation')
    ax.set_xlabel('Epoch', fontsize=13, fontweight='medium')
    ax.set_ylabel('MSE Loss', fontsize=13, fontweight='medium')
    ax.set_title('MLP', fontsize=15, fontweight='bold', pad=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.tick_params(axis='both', labelsize=11)
    
    fig.suptitle('Training Progress: Convex Hull Volume Prediction',
                 fontsize=16, fontweight='bold', y=1.0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved training curves to {save_path}")
    
    return fig, axes


def plot_layerwise_bar(avg_losses, std_losses, num_phi_layers, num_rho_layers, 
                       title='SO(3) Symmetry Emergence Through Network Depth',
                       save_path='convex_hull_layerwise.png'):
    """
    Create a bar chart showing symmetry loss per layer with color-coded regions.
    
    Args:
        avg_losses: dict mapping layer indices to average symmetry losses
        std_losses: dict mapping layer indices to standard deviations
        num_phi_layers: Number of phi layers
        num_rho_layers: Number of rho layers
        title: Plot title
        save_path: Path to save the figure
    """
    # Build layer names and values in order
    layer_names = []
    layer_values = []
    layer_stds = []
    layer_colors = []
    
    # Color scheme
    color_phi = '#3b9ddd'    # Blue for phi (per-particle)
    color_pool = '#22c55e'   # Green for pool
    color_rho = '#f97066'    # Coral for rho (post-pool)
    color_output = '#a78bfa' # Purple for output
    
    # Phi layers (1 to num_phi_layers)
    for i in range(1, num_phi_layers + 1):
        layer_names.append(f'phi_{i}')
        layer_values.append(avg_losses[i])
        layer_stds.append(std_losses[i])
        layer_colors.append(color_phi)
    
    # Pool layer (num_phi_layers + 1)
    pool_idx = num_phi_layers + 1
    layer_names.append('pool')
    layer_values.append(avg_losses[pool_idx])
    layer_stds.append(std_losses[pool_idx])
    layer_colors.append(color_pool)
    
    # Rho layers (num_phi_layers + 2 onwards)
    for i in range(1, num_rho_layers + 1):
        rho_idx = num_phi_layers + 1 + i
        layer_names.append(f'rho_{i}')
        layer_values.append(avg_losses[rho_idx])
        layer_stds.append(std_losses[rho_idx])
        layer_colors.append(color_rho)
    
    # Output layer (-1)
    layer_names.append('output')
    layer_values.append(avg_losses[-1])
    layer_stds.append(std_losses[-1])
    layer_colors.append(color_output)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#f8f9fa')
    
    x_pos = np.arange(len(layer_names))
    
    # Create bars
    bars = ax.bar(x_pos, layer_values, yerr=layer_stds, capsize=4,
                  color=layer_colors, edgecolor='black', linewidth=0.5,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    
    # Styling
    ax.set_xlabel('Layer', fontsize=14, fontweight='medium')
    ax.set_ylabel('Relative Symmetry Loss (log scale)', fontsize=14, fontweight='medium')
    ax.set_title(f'{title}\n(Lower = More Invariant)', fontsize=16, fontweight='bold', pad=15)
    
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, fontsize=11, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=11)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_phi, edgecolor='black', linewidth=0.5, label='Phi (per-particle)'),
        Patch(facecolor=color_pool, edgecolor='black', linewidth=0.5, label='Pool'),
        Patch(facecolor=color_rho, edgecolor='black', linewidth=0.5, label='Rho (post-pool)'),
        Patch(facecolor=color_output, edgecolor='black', linewidth=0.5, label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved layerwise bar plot to {save_path}")
    
    return fig, ax


def plot_comparison(deepsets_results, mlp_results, num_epochs, save_path='convex_hull_symmetry.png'):
    """
    Create side-by-side visualization comparing symmetry emergence in both architectures.
    
    Args:
        deepsets_results: List of (epoch, avg_losses, std_losses) for DeepSets
        mlp_results: List of (epoch, avg_losses, std_losses) for MLP
        num_epochs: Total number of training epochs
        save_path: Path to save the figure
    """
    # Color scheme
    colors = ['#94a3b8', '#64748b', '#475569', '#1e293b']
    markers = ['o', 's', '^', 'D']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot DeepSets
    ax = axes[0]
    ax.set_facecolor('#f8f9fa')
    
    # Get DeepSets layer info
    ds_first = deepsets_results[0]
    ds_layer_indices = [k for k in ds_first[1].keys() if k != -1]
    ds_layer_indices = sorted(ds_layer_indices) + [-1]
    
    # Determine phi/rho boundary (assuming first result has model info)
    # For now, create x labels based on layer count
    num_phi = len([k for k in ds_layer_indices if k != -1 and k <= len(ds_layer_indices)//2])
    
    x_positions = list(range(len(ds_layer_indices)))
    x_labels = []
    for i, idx in enumerate(ds_layer_indices):
        if idx == -1:
            x_labels.append('out')
        else:
            x_labels.append(f'{idx}')
    
    for i, (epoch, avg_losses, std_losses) in enumerate(deepsets_results):
        losses = [avg_losses[idx] for idx in ds_layer_indices]
        stds = [std_losses[idx] for idx in ds_layer_indices]
        
        if epoch == 0:
            label = 'Epoch 0 (untrained)'
        elif epoch == num_epochs:
            label = f'Epoch {epoch} (final)'
        else:
            fraction = epoch / num_epochs
            label = f'Epoch {epoch} ({fraction:.0%} trained)'
        
        ax.errorbar(x_positions, losses, yerr=stds,
                    fmt=f'{markers[i]}-', capsize=4, capthick=1.5,
                    color=colors[i], linewidth=2, markersize=9,
                    markerfacecolor=colors[i], markeredgecolor='white',
                    markeredgewidth=1.5, ecolor=colors[i], alpha=0.9,
                    label=label)
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='medium')
    ax.set_ylabel('Relative Symmetry Loss', fontsize=14, fontweight='medium')
    ax.set_title('DeepSets (Permutation Invariant)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    # Plot MLP
    ax = axes[1]
    ax.set_facecolor('#f8f9fa')
    
    mlp_first = mlp_results[0]
    mlp_layer_indices = [k for k in mlp_first[1].keys() if k != -1]
    mlp_layer_indices = sorted(mlp_layer_indices) + [-1]
    
    x_positions = list(range(len(mlp_layer_indices)))
    x_labels = []
    for idx in mlp_layer_indices:
        if idx == -1:
            x_labels.append('out')
        else:
            x_labels.append(f'{idx}')
    
    for i, (epoch, avg_losses, std_losses) in enumerate(mlp_results):
        losses = [avg_losses[idx] for idx in mlp_layer_indices]
        stds = [std_losses[idx] for idx in mlp_layer_indices]
        
        if epoch == 0:
            label = 'Epoch 0 (untrained)'
        elif epoch == num_epochs:
            label = f'Epoch {epoch} (final)'
        else:
            fraction = epoch / num_epochs
            label = f'Epoch {epoch} ({fraction:.0%} trained)'
        
        ax.errorbar(x_positions, losses, yerr=stds,
                    fmt=f'{markers[i]}-', capsize=4, capthick=1.5,
                    color=colors[i], linewidth=2, markersize=9,
                    markerfacecolor=colors[i], markeredgecolor='white',
                    markeredgewidth=1.5, ecolor=colors[i], alpha=0.9,
                    label=label)
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='medium')
    ax.set_ylabel('Relative Symmetry Loss', fontsize=14, fontweight='medium')
    ax.set_title('MLP (Flattened Input)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    # Overall title
    fig.suptitle('SO(3) Symmetry Emergence: Convex Hull Volume Prediction\n(Baseline: No Symmetry Penalty)',
                 fontsize=18, fontweight='bold', y=1.0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to {save_path}")
    
    return fig, axes


def main(n_samples=10000, n_points=50, learning_rate=3e-4, num_epochs=100, 
         run_seed=42, save_path='convex_hull_symmetry.png'):
    """
    Main function to train both architectures and analyze layerwise symmetry.
    
    Args:
        n_samples: Number of samples in dataset
        n_points: Number of points per sample
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        run_seed: Random seed for reproducibility
        save_path: Path to save the visualization
    """
    print(f"Device: {device}")
    print(f"Dataset: {n_samples} samples, {n_points} points each")
    print(f"Training for {num_epochs} epochs with lr={learning_rate}")
    print()
    
    # Derive seeds
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    
    # Create dataset
    print("Generating convex hull dataset...")
    dataset = ConvexHullDataset(n_samples, n_points=n_points, seed=data_seed)
    print(f"Volume range: [{dataset.volumes.min().item():.2f}, {dataset.volumes.max().item():.2f}]")
    print(f"Volume mean: {dataset.volume_mean:.2f}, std: {dataset.volume_std:.2f}")
    
    # Create flattened version for MLP
    dataset_flat = ConvexHullDatasetFlattened(dataset)
    
    batch_size = 128
    generator = torch.Generator().manual_seed(data_seed)
    
    # Split data
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [0.6, 0.2, 0.2], generator=generator
    )
    train_ds_flat, val_ds_flat, test_ds_flat = torch.utils.data.random_split(
        dataset_flat, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(data_seed)
    )
    
    loader_generator = torch.Generator().manual_seed(data_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    loader_generator_flat = torch.Generator().manual_seed(data_seed)
    train_loader_flat = DataLoader(train_ds_flat, batch_size=batch_size, shuffle=True, generator=loader_generator_flat)
    val_loader_flat = DataLoader(val_ds_flat, batch_size=batch_size, shuffle=False)
    test_loader_flat = DataLoader(test_ds_flat, batch_size=batch_size, shuffle=False)
    
    # ============================================================
    # Train DeepSets
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING DEEPSETS")
    print("="*60)
    
    set_model_seed(model_seed)
    # Architecture: phi [3, 64, 64], rho [64, 32, 1]
    # Total layers: 2 (phi) + 1 (pool) + 2 (rho) = 5
    deepsets = DeepSets(phi_dims=[3, 64, 64, 64, 64], rho_dims=[64, 64, 64, 32, 1], pooling='sum').to(device)
    print(f"DeepSets: phi layers={deepsets.num_phi_linear_layers}, rho layers={deepsets.num_rho_linear_layers}")
    print(f"Total parameters: {sum(p.numel() for p in deepsets.parameters()):,}")
    
    optimizer = torch.optim.AdamW(deepsets.parameters(), lr=learning_rate, weight_decay=0.0)
    deepsets_checkpoints, deepsets_train_losses, deepsets_val_losses = train_with_checkpoints(
        deepsets, train_loader, val_loader, optimizer,
        num_epochs=num_epochs, model_type='deepsets'
    )
    
    # Compute layerwise symmetry for DeepSets
    print("\nAnalyzing DeepSets symmetry at each checkpoint...")
    deepsets_results = []
    for epoch, state_dict in deepsets_checkpoints:
        print(f"  Evaluating checkpoint at epoch {epoch}...")
        deepsets.load_state_dict(state_dict)
        avg_losses, std_losses = compute_layerwise_symmetry_deepsets(
            deepsets, test_loader, num_samples=10, show_progress=False
        )
        deepsets_results.append((epoch, avg_losses, std_losses))
        print(f"    Layer 1: {avg_losses[1]:.4f}, Output: {avg_losses[-1]:.4f}")
    
    # ============================================================
    # Train MLP
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING MLP")
    print("="*60)
    
    set_model_seed(model_seed)
    # Match similar capacity: [150, 128, 64, 32, 1]
    mlp_dims = [n_points * 3, 128, 64, 32, 1]
    mlp = MLP(mlp_dims).to(device)
    print(f"MLP: dims={mlp_dims}")
    print(f"Total parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=learning_rate, weight_decay=0.0)
    mlp_checkpoints, mlp_train_losses, mlp_val_losses = train_with_checkpoints(
        mlp, train_loader_flat, val_loader_flat, optimizer,
        num_epochs=num_epochs, model_type='mlp'
    )
    
    # Compute layerwise symmetry for MLP
    print("\nAnalyzing MLP symmetry at each checkpoint...")
    mlp_results = []
    for epoch, state_dict in mlp_checkpoints:
        print(f"  Evaluating checkpoint at epoch {epoch}...")
        mlp.load_state_dict(state_dict)
        avg_losses, std_losses = compute_layerwise_symmetry_mlp(
            mlp, test_loader_flat, n_points, num_samples=10, show_progress=False
        )
        mlp_results.append((epoch, avg_losses, std_losses))
        print(f"    Layer 1: {avg_losses[1]:.4f}, Output: {avg_losses[-1]:.4f}")
    
    # ============================================================
    # Print summary
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY: DEEPSETS LAYERWISE SYMMETRY LOSS")
    print("="*60)
    ds_layer_indices = sorted([k for k in deepsets_results[0][1].keys() if k != -1]) + [-1]
    
    header = "Epoch".ljust(12) + "".join([f"L{i}".rjust(10) for i in ds_layer_indices if i != -1]) + "Output".rjust(10)
    print(header)
    print("-" * len(header))
    
    for epoch, avg_losses, _ in deepsets_results:
        row = f"{epoch}".ljust(12)
        for idx in ds_layer_indices:
            row += f"{avg_losses[idx]:.4f}".rjust(10)
        print(row)
    
    print("\n" + "="*60)
    print("SUMMARY: MLP LAYERWISE SYMMETRY LOSS")
    print("="*60)
    mlp_layer_indices = sorted([k for k in mlp_results[0][1].keys() if k != -1]) + [-1]
    
    header = "Epoch".ljust(12) + "".join([f"L{i}".rjust(10) for i in mlp_layer_indices if i != -1]) + "Output".rjust(10)
    print(header)
    print("-" * len(header))
    
    for epoch, avg_losses, _ in mlp_results:
        row = f"{epoch}".ljust(12)
        for idx in mlp_layer_indices:
            row += f"{avg_losses[idx]:.4f}".rjust(10)
        print(row)
    
    # Generate visualizations
    # Training curves
    base, ext = os.path.splitext(save_path)
    training_save_path = f"{base}_training{ext}"
    plot_training_curves(deepsets_train_losses, deepsets_val_losses,
                         mlp_train_losses, mlp_val_losses, training_save_path)
    
    # Layerwise bar plot for final DeepSets checkpoint
    final_epoch, final_avg_losses, final_std_losses = deepsets_results[-1]
    layerwise_save_path = f"{base}_layerwise{ext}"
    plot_layerwise_bar(final_avg_losses, final_std_losses,
                       num_phi_layers=deepsets.num_phi_linear_layers,
                       num_rho_layers=deepsets.num_rho_linear_layers,
                       title='SO(3) Symmetry Emergence Through Network Depth',
                       save_path=layerwise_save_path)
    
    # Symmetry analysis (line plot over training)
    fig, axes = plot_comparison(deepsets_results, mlp_results, num_epochs, save_path)
    plt.show()
    
    return deepsets_results, mlp_results, (deepsets_train_losses, deepsets_val_losses, mlp_train_losses, mlp_val_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze SO(3) symmetry emergence in convex hull prediction')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples in dataset (default: 10000)')
    parser.add_argument('--n-points', type=int, default=50,
                        help='Number of points per sample (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for optimizer (default: 3e-4)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--run-seed', type=int, default=42,
                        help='Base random seed for reproducibility (default: 42)')
    parser.add_argument('--save-path', type=str, default='convex_hull_symmetry.png',
                        help='Path to save the visualization (default: convex_hull_symmetry.png)')
    
    args = parser.parse_args()
    
    main(
        n_samples=args.n_samples,
        n_points=args.n_points,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        run_seed=args.run_seed,
        save_path=args.save_path
    )

