import warnings

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .train import plot_training_loss


def plot_regression_surface(
    model, 
    dataset, 
    save_path, 
    device,
    field_name: str = None,
    sym_penalty_type: str = None,
    sym_layers: list = None,
    lambda_sym: float = 0.0,
):
    """
    Plot predicted vs true regression surface.
    
    Creates a 3-panel figure showing:
    - Predicted field
    - True field  
    - Error (predicted - true)
    
    Args:
        model: Trained neural network model.
        dataset: Dataset with r_min, r_max, and scalar_field_fn attributes.
        save_path: Path to save the figure.
        device: Torch device.
        field_name: Name of the scalar field used for training.
        sym_penalty_type: Type of symmetry penalty used during training.
        sym_layers: List of layers penalized during training.
        lambda_sym: Lambda value for symmetry penalty.
    """
    model.eval()
    model.to(device)
    
    r_min, r_max = dataset.r_min, dataset.r_max
    margin = r_max * 0.1
    extent = r_max + margin
    
    xx, yy = np.meshgrid(np.linspace(-extent, extent, 200), np.linspace(-extent, extent, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(grid).cpu().numpy().reshape(xx.shape)
    
    true_radii = np.sqrt(xx**2 + yy**2)
    true_field = dataset.scalar_field_fn(xx, yy, true_radii)
    
    # Mask out-of-distribution points (outside the disk)
    mask = (true_radii < r_min) | (true_radii > r_max)
    preds = np.ma.masked_where(mask, preds)
    true_field = np.ma.masked_where(mask, true_field)
    error = np.ma.masked_where(mask, preds - true_field)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Build suptitle with field name and training info
    suptitle_lines = []
    if field_name:
        suptitle_lines.append(f'Field = {field_name}')
    if sym_penalty_type and lambda_sym > 0 and sym_layers:
        layers_str = str(sym_layers).replace(' ', '')
        suptitle_lines.append(f'Model trained with {sym_penalty_type} penalty (layers={layers_str}, λ={lambda_sym})')
    elif field_name:
        suptitle_lines.append('Model trained without symmetry penalty')
    
    if suptitle_lines:
        fig.suptitle('\n'.join(suptitle_lines), fontsize=11, fontweight='bold')
    
    vmin, vmax = true_field.min(), true_field.max()
    for ax, data, title in zip(axes, [preds, true_field, error], 
                                ['Predicted', 'True', 'Error']):
        if title == 'Error':
            max_err = max(abs(data.min()), abs(data.max()), 0.1)
            cf = ax.contourf(xx, yy, data, levels=50, cmap='RdBu_r', vmin=-max_err, vmax=max_err)
            ax.contour(xx, yy, data, levels=[0], colors='black', linewidths=1)
        else:
            cf = ax.contourf(xx, yy, data, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.contour(xx, yy, data, levels=np.linspace(vmin, vmax, 11), colors='white', linewidths=0.5, alpha=0.7)
        plt.colorbar(cf, ax=ax)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_run_summary(
    history: dict,
    Q_values: dict,
    oracle_Q: float,
    model: nn.Module,
    dataset,
    device: torch.device,
    save_path: Path,
    run_name: str = None,
    MI_values: dict = None,
    oracle_MI: float = None,
    xlim: tuple = None,
    lambda_sym: float = 0.0,
    field_name: str = None,
    sym_penalty_type: str = None,
    sym_layers: list = None,
    Q_stds: dict = None,
):
    """
    Create a combined summary plot with loss curves, Q/MI metrics, and regression surface.
    
    Mosaic layout:
        AACCC
        BBCCC
    
    Where:
        A = Loss plot (spans 2 columns)
        B = Q and MI metrics (grouped bar chart, spans 2 columns)
        C = Regression surface panels (2x2 grid with 3 panels + 1 empty, wider aspect)
    
    Args:
        history: Training history dict with 'step', 'train_loss', 'val_loss', etc.
        Q_values: Dictionary mapping layer names to Q values.
        oracle_Q: Oracle Q value.
        model: Trained model.
        dataset: Dataset with r_min, r_max, scalar_field_fn attributes.
        device: Torch device.
        save_path: Path to save the combined figure.
        run_name: Optional run name for title.
        MI_values: Optional dictionary mapping layer names to MI values.
        oracle_MI: Optional oracle MI value.
        xlim: Optional tuple (xmin, xmax) for fixed x-axis limits on loss plot.
              Useful for creating animation frames with consistent axes.
        lambda_sym: Lambda weight for symmetry penalty (for plotting scaled sym loss).
        field_name: Name of the scalar field used for training.
        sym_penalty_type: Type of symmetry penalty used during training.
        sym_layers: List of layers penalized during training.
        Q_stds: Optional dictionary mapping layer names to Q standard errors for error bars.
    """
    # Create figure with mosaic layout: 2 rows x 4 columns with width ratios
    fig = plt.figure(figsize=(16, 9))
    
    # Build suptitle with field name and training info
    suptitle_lines = []
    if field_name:
        suptitle_lines.append(f'Field = {field_name}')
    if sym_penalty_type and lambda_sym > 0 and sym_layers:
        layers_str = str(sym_layers).replace(' ', '')
        suptitle_lines.append(f'Model trained with {sym_penalty_type} penalty (layers={layers_str}, λ={lambda_sym})')
    elif field_name:
        suptitle_lines.append('Model trained without symmetry penalty')
    
    # Add run_name if provided and we have a suptitle
    if run_name and suptitle_lines:
        suptitle_lines.append(run_name)
    elif run_name:
        suptitle_lines = [run_name]
    
    suptitle = '\n'.join(suptitle_lines) if suptitle_lines else 'Run Summary'
    fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.98)
    
    # Create grid layout with width_ratios: A/B get 2 units, C area gets 3 units (1.5+1.5)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3, width_ratios=[1, 1, 1.5, 1.5])
    
    # === A: Loss curves (top left, spans cols 0-1) ===
    ax_loss = fig.add_subplot(gs[0, :2])
    
    # Use shared plotting function
    plot_training_loss(
        ax_loss, history, lambda_sym=lambda_sym, xlim=xlim,
        title='Training and Validation Loss'
    )
    
    # === B: Q and MI metrics (bottom left, spans cols 0-1) ===
    ax_metrics = fig.add_subplot(gs[1, :2])
    
    # Build layer list (without oracle for now)
    layers = list(Q_values.keys())
    q_vals = list(Q_values.values())
    
    # Get Q standard errors if provided
    has_q_stds = Q_stds is not None and len(Q_stds) > 0
    if has_q_stds:
        q_err = [Q_stds.get(layer, 0) for layer in layers]
    else:
        q_err = None
    
    # Get MI values if provided
    has_mi = MI_values is not None and len(MI_values) > 0
    if has_mi:
        mi_vals = [MI_values.get(layer, 0) for layer in layers]
    
    # Add oracle values
    if oracle_Q is not None:
        layers = layers + ['oracle']
        q_vals = q_vals + [oracle_Q]
        if has_q_stds:
            q_err = q_err + [0]  # No error bar for oracle (deterministic)
        if has_mi:
            mi_vals = mi_vals + [oracle_MI if oracle_MI is not None else 0]
    
    x = np.arange(len(layers))
    
    if has_mi:
        # Grouped bar chart with Q and MI
        width = 0.35
        bars_q = ax_metrics.bar(x - width/2, q_vals, width, label='Q', color='steelblue', edgecolor='black',
                                 yerr=q_err if has_q_stds else None, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        bars_mi = ax_metrics.bar(x + width/2, mi_vals, width, label='MI', color='darkorange', edgecolor='black')
        
        # Color oracle bars green
        if oracle_Q is not None:
            bars_q[-1].set_color('green')
            bars_mi[-1].set_color('forestgreen')
        
        ax_metrics.legend(fontsize='small', loc='upper right')
        ax_metrics.set_title('SO(2) Invariance Metrics by Layer (Q and MI)')
    else:
        # Just Q bars (original behavior)
        colors = ['steelblue'] * (len(layers) - 1) + ['green'] if oracle_Q is not None else ['steelblue'] * len(layers)
        ax_metrics.bar(x, q_vals, color=colors, edgecolor='black',
                       yerr=q_err if has_q_stds else None, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        ax_metrics.set_title('SO(2) Invariance Metric by Layer')
    
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(layers, rotation=45, ha='right', fontsize='small')
    ax_metrics.set_xlabel('Layer')
    ax_metrics.set_ylabel('Metric Value')
    ax_metrics.set_ylim(bottom=0)
    ax_metrics.grid(axis='y', alpha=0.3)
    
    # === C: Regression surfaces (2x2 grid in cols 2-3) ===
    model.eval()
    model.to(device)
    
    r_min, r_max = dataset.r_min, dataset.r_max
    margin = r_max * 0.1
    extent = r_max + margin
    
    xx, yy = np.meshgrid(np.linspace(-extent, extent, 200), np.linspace(-extent, extent, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(grid).cpu().numpy().reshape(xx.shape)
    
    true_radii = np.sqrt(xx**2 + yy**2)
    true_field = dataset.scalar_field_fn(xx, yy, true_radii)
    
    # Mask out-of-distribution points
    mask = (true_radii < r_min) | (true_radii > r_max)
    preds_masked = np.ma.masked_where(mask, preds)
    true_masked = np.ma.masked_where(mask, true_field)
    error_masked = np.ma.masked_where(mask, preds - true_field)
    
    # Compute orbit-averaged prediction (average over angles for each radius)
    n_radii = 100
    n_angles = 64
    radii_samples = np.linspace(r_min, r_max, n_radii)
    angles_samples = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    
    orbit_means = []
    with torch.no_grad():
        for r in radii_samples:
            # Sample points around this orbit
            xs = r * np.cos(angles_samples)
            ys = r * np.sin(angles_samples)
            orbit_pts = torch.tensor(np.c_[xs, ys], dtype=torch.float32).to(device)
            orbit_preds = model(orbit_pts).cpu().numpy().flatten()
            orbit_means.append(orbit_preds.mean())
    orbit_means = np.array(orbit_means)
    
    # Compute true radial function for comparison
    true_radial = dataset.scalar_field_fn(radii_samples, np.zeros_like(radii_samples), radii_samples)
    
    # Compute area-weighted mean of true function (with respect to disk measure: r dr dθ)
    # Mean = ∫ f(r) r dr / ∫ r dr
    true_mean = np.trapezoid(true_radial * radii_samples, radii_samples) / np.trapezoid(radii_samples, radii_samples)
    
    vmin, vmax = true_masked.min(), true_masked.max()
    
    # C panels in 2x2 arrangement: (0,2)=Predicted, (0,3)=True, (1,2)=Orbit Avg, (1,3)=Error
    
    # Predicted (top left of C area)
    ax_pred = fig.add_subplot(gs[0, 2])
    cf_pred = ax_pred.contourf(xx, yy, preds_masked, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cf_pred, ax=ax_pred)
    ax_pred.set_xlim(-extent, extent)
    ax_pred.set_ylim(-extent, extent)
    ax_pred.set_xlabel('x')
    ax_pred.set_ylabel('y')
    ax_pred.set_title('Predicted')
    ax_pred.set_aspect('equal')
    
    # True (top right of C area)
    ax_true = fig.add_subplot(gs[0, 3])
    cf_true = ax_true.contourf(xx, yy, true_masked, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cf_true, ax=ax_true)
    ax_true.set_xlim(-extent, extent)
    ax_true.set_ylim(-extent, extent)
    ax_true.set_xlabel('x')
    ax_true.set_ylabel('y')
    ax_true.set_title('True')
    ax_true.set_aspect('equal')
    
    # Orbit Average as 1D radial plot (bottom left of C area)
    ax_orbit = fig.add_subplot(gs[1, 2])
    ax_orbit.plot(radii_samples, true_radial, 'k-', linewidth=2, label='True f(r)')
    ax_orbit.plot(radii_samples, orbit_means, 'b-', linewidth=2, label='Orbit Avg')
    ax_orbit.axhline(y=true_mean, color='gray', linestyle='--', linewidth=1.5, label=f'Mean = {true_mean:.3f}')
    ax_orbit.set_xlabel('r')
    ax_orbit.set_ylabel('f(r)')
    ax_orbit.set_title('Radial Function')
    ax_orbit.legend(fontsize='small')
    ax_orbit.grid(True, alpha=0.3)
    
    # Error (bottom right of C area)
    ax_err = fig.add_subplot(gs[1, 3])
    max_err = max(abs(error_masked.min()), abs(error_masked.max()), 0.1)
    cf_err = ax_err.contourf(xx, yy, error_masked, levels=50, cmap='RdBu_r', vmin=-max_err, vmax=max_err)
    plt.colorbar(cf_err, ax=ax_err)
    ax_err.set_xlim(-extent, extent)
    ax_err.set_ylim(-extent, extent)
    ax_err.set_xlabel('x')
    ax_err.set_ylabel('y')
    ax_err.set_title('Error')
    ax_err.set_aspect('equal')
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
