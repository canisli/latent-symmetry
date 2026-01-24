import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_regression_surface(model, dataset, save_path, device):
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
):
    """
    Create a combined summary plot with loss curves, Q metric, and regression surface.
    
    Mosaic layout:
        AACCC
        BBCCC
    
    Where:
        A = Loss plot (spans 2 columns)
        B = Q plot (spans 2 columns)
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
    """
    # Create figure with mosaic layout: 2 rows x 4 columns with width ratios
    fig = plt.figure(figsize=(16, 8))
    
    # Title
    title = run_name if run_name else 'Run Summary'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Create grid layout with width_ratios: A/B get 2 units, C area gets 3 units (1.5+1.5)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3, width_ratios=[1, 1, 1.5, 1.5])
    
    # === A: Loss curves (top left, spans cols 0-1) ===
    ax_loss = fig.add_subplot(gs[0, :2])
    if history['step']:  # Only plot if training happened
        steps = np.array(history['step'])
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        steps_per_epoch = max(1, int(history.get('steps_per_epoch', 1)))
        
        epoch_idx = (steps - 1) // steps_per_epoch
        epoch_vals = np.unique(epoch_idx)
        epoch_train = np.array([train_loss[epoch_idx == e].mean() for e in epoch_vals])
        epoch_val = np.array([val_loss[epoch_idx == e].mean() for e in epoch_vals])
        
        ax_loss.plot(steps, train_loss, color='tab:blue', alpha=0.25, linewidth=1, label='Train (step)')
        ax_loss.plot(steps, val_loss, color='tab:orange', alpha=0.25, linewidth=1, label='Val (step)')
        ax_loss.plot((epoch_vals + 1) * steps_per_epoch, epoch_train, color='tab:blue', linewidth=2, label='Train (epoch)')
        ax_loss.plot((epoch_vals + 1) * steps_per_epoch, epoch_val, color='tab:orange', linewidth=2, label='Val (epoch)')
        ax_loss.set_yscale('log')
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize='small')
        ax_loss.grid(True, alpha=0.3)
    else:
        ax_loss.text(0.5, 0.5, 'No training (steps=0)', ha='center', va='center', transform=ax_loss.transAxes)
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training and Validation Loss')
    
    # === B: Q metric (bottom left, spans cols 0-1) ===
    ax_q = fig.add_subplot(gs[1, :2])
    layers = list(Q_values.keys())
    values = list(Q_values.values())
    if oracle_Q is not None:
        layers = layers + ['oracle']
        values = values + [oracle_Q]
    
    x = range(len(layers))
    colors = ['steelblue'] * (len(layers) - 1) + ['green'] if oracle_Q is not None else ['steelblue'] * len(layers)
    
    ax_q.bar(x, values, color=colors, edgecolor='black')
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(layers, rotation=45, ha='right', fontsize='small')
    ax_q.set_xlabel('Layer')
    ax_q.set_ylabel('Q')
    ax_q.set_title('SO(2) Invariance Metric by Layer')
    ax_q.set_ylim(bottom=0)
    ax_q.grid(axis='y', alpha=0.3)
    
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
    
    vmin, vmax = true_masked.min(), true_masked.max()
    
    # C panels in 2x2 arrangement: (0,2)=Predicted, (0,3)=True, (1,2)=Error, (1,3)=empty
    
    # Predicted (top middle)
    ax_pred = fig.add_subplot(gs[0, 2])
    cf_pred = ax_pred.contourf(xx, yy, preds_masked, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cf_pred, ax=ax_pred)
    ax_pred.set_xlim(-extent, extent)
    ax_pred.set_ylim(-extent, extent)
    ax_pred.set_xlabel('x')
    ax_pred.set_ylabel('y')
    ax_pred.set_title('Predicted')
    ax_pred.set_aspect('equal')
    
    # True (top right)
    ax_true = fig.add_subplot(gs[0, 3])
    cf_true = ax_true.contourf(xx, yy, true_masked, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cf_true, ax=ax_true)
    ax_true.set_xlim(-extent, extent)
    ax_true.set_ylim(-extent, extent)
    ax_true.set_xlabel('x')
    ax_true.set_ylabel('y')
    ax_true.set_title('True')
    ax_true.set_aspect('equal')
    
    # Error (bottom middle)
    ax_err = fig.add_subplot(gs[1, 2])
    max_err = max(abs(error_masked.min()), abs(error_masked.max()), 0.1)
    cf_err = ax_err.contourf(xx, yy, error_masked, levels=50, cmap='RdBu_r', vmin=-max_err, vmax=max_err)
    plt.colorbar(cf_err, ax=ax_err)
    ax_err.set_xlim(-extent, extent)
    ax_err.set_ylim(-extent, extent)
    ax_err.set_xlabel('x')
    ax_err.set_ylabel('y')
    ax_err.set_title('Error')
    ax_err.set_aspect('equal')
    
    # (1,3) is intentionally left empty
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
