"""
Training utilities for radius regression.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Tuple
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from .symmetry_penalty import SymmetryPenalty


def train_step(
    model: nn.Module,
    batch: tuple,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    symmetry_penalty: Optional[SymmetryPenalty] = None,
    lambda_sym: float = 0.0,
    sym_layers: Optional[List[int]] = None,
    n_augmentations: int = 4,
    compute_grad_alignment: bool = False,
) -> Tuple[float, float, Optional[float]]:
    """
    Single training step with optional symmetry penalty.
    
    Args:
        model: Neural network model.
        batch: Tuple of (X, y) tensors.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: Device to use.
        symmetry_penalty: Optional symmetry penalty instance.
        lambda_sym: Weight for symmetry penalty (0 = disabled).
        sym_layers: List of layer indices to penalize.
        n_augmentations: Number of rotation pairs per sample.
        compute_grad_alignment: If True, compute cosine similarity between
            task and symmetry gradients (slower, for diagnostics).
    
    Returns:
        Tuple of (task_loss, sym_loss, grad_cosine_sim) as floats.
        grad_cosine_sim is None if compute_grad_alignment=False or no sym penalty.
    """
    model.train()
    X, y = batch
    X, y = X.to(device), y.to(device)
    
    optimizer.zero_grad()
    preds = model(X)
    task_loss = loss_fn(preds, y)
    
    sym_loss_value = 0.0
    grad_cosine_sim = None
    
    if symmetry_penalty is not None and lambda_sym > 0 and sym_layers:
        sym_loss = symmetry_penalty.compute_total(
            model, X, sym_layers, n_augmentations, device
        )
        sym_loss_value = sym_loss.item()
        
        if compute_grad_alignment:
            # Compute task gradient
            task_loss.backward(retain_graph=True)
            task_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            
            # Compute symmetry gradient
            optimizer.zero_grad()
            sym_loss.backward(retain_graph=True)
            sym_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            
            # Cosine similarity
            task_norm = task_grad.norm()
            sym_norm = sym_grad.norm()
            if task_norm > 0 and sym_norm > 0:
                grad_cosine_sim = (task_grad @ sym_grad / (task_norm * sym_norm)).item()
            else:
                grad_cosine_sim = 0.0
            
            # Now compute total gradient for actual update
            optimizer.zero_grad()
        
        total_loss = task_loss + lambda_sym * sym_loss
    else:
        total_loss = task_loss
    
    total_loss.backward()
    optimizer.step()
    
    return task_loss.item(), sym_loss_value, grad_cosine_sim


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.
    
    Args:
        model: Neural network model.
        dataloader: DataLoader for evaluation.
        loss_fn: Loss function.
        device: Device to use.
        desc: Description for progress bar.
    
    Returns:
        Dictionary with 'loss' (MSE) and 'mae' (mean absolute error) keys.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total = 0
    
    for X, y in tqdm(dataloader, desc=desc, leave=False):
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)
        total_loss += loss.item() * X.size(0)
        
        # Compute MAE
        total_mae += torch.abs(preds - y).sum().item()
        total += X.size(0)
    
    return {
        'loss': total_loss / total,
        'mae': total_mae / total,
    }


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 100,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create cosine annealing scheduler with warmup.
    
    Args:
        optimizer: Optimizer.
        total_steps: Total training steps.
        warmup_steps: Number of warmup steps.
    
    Returns:
        Learning rate scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    total_steps: int,
    log_interval: int = 100,
    eval_interval: int = 500,
    save_dir: Optional[Path] = None,
    save_best: bool = True,
    symmetry_penalty: Optional[SymmetryPenalty] = None,
    lambda_sym: float = 0.0,
    sym_layers: Optional[List[int]] = None,
    n_augmentations: int = 4,
    frame_callback: Optional[Callable[[int, nn.Module, Dict], None]] = None,
    frame_interval: int = 10,
    grad_align_interval: int = 0,
) -> Dict[str, list]:
    """
    Main training loop for regression with optional symmetry penalty.
    
    Args:
        model: Neural network model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        loss_fn: Loss function (e.g., MSELoss).
        optimizer: Optimizer.
        scheduler: Optional learning rate scheduler.
        device: Device to use.
        total_steps: Total training steps.
        log_interval: Log every N steps.
        eval_interval: Evaluate every N steps.
        save_dir: Directory to save checkpoints.
        save_best: Whether to save best model.
        symmetry_penalty: Optional symmetry penalty instance.
        lambda_sym: Weight for symmetry penalty (0 = disabled).
        sym_layers: List of layer indices to penalize.
        n_augmentations: Number of rotation pairs per sample.
        frame_callback: Optional callback(step, model, history) for dynamics visualization.
        frame_interval: Steps between frame_callback invocations.
        grad_align_interval: Compute gradient alignment every N steps (0 = disabled).
    
    Returns:
        Dictionary of training metrics history.
    """
    model.to(device)
    
    history = {
        'batch_step': [],         # Step number for each batch
        'batch_loss': [],         # Per-batch task loss (training dynamics)
        'batch_sym_loss': [],     # Per-batch symmetry loss
        'grad_align_step': [],    # Steps where gradient alignment was computed
        'grad_align': [],         # Cosine similarity between task and sym gradients
        'eval_step': [],          # Step number for each evaluation
        'train_loss': [],         # Full-dataset task loss evaluation
        'val_loss': [],
        'train_sym_loss': [],     # Full-dataset symmetry loss on train set
        'val_sym_loss': [],       # Full-dataset symmetry loss on val set
        'val_mae': [],
        'lr': [],
    }
    steps_per_epoch = len(train_loader)
    
    best_val_loss = float('inf')
    step = 0
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:
        # Get next batch (cycle through dataloader)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Decide whether to compute gradient alignment this step
        compute_grad_align = (
            grad_align_interval > 0 and 
            (step + 1) % grad_align_interval == 0 and
            symmetry_penalty is not None and 
            lambda_sym > 0 and 
            sym_layers
        )
        
        # Training step
        task_loss, sym_loss, grad_cos = train_step(
            model, batch, loss_fn, optimizer, device,
            symmetry_penalty=symmetry_penalty,
            lambda_sym=lambda_sym,
            sym_layers=sym_layers,
            n_augmentations=n_augmentations,
            compute_grad_alignment=compute_grad_align,
        )
        
        # Record per-batch losses
        history['batch_step'].append(step + 1)
        history['batch_loss'].append(task_loss)
        history['batch_sym_loss'].append(sym_loss)
        
        # Record gradient alignment if computed
        if grad_cos is not None:
            history['grad_align_step'].append(step + 1)
            history['grad_align'].append(grad_cos)
        
        if scheduler is not None:
            scheduler.step()
        
        # Frame callback for dynamics visualization
        if frame_callback is not None and (step + 1) % frame_interval == 0:
            frame_callback(step + 1, model, history)
        
        # Evaluation
        if step % eval_interval == 0 or step == total_steps - 1:
            val_metrics = evaluate(model, val_loader, loss_fn, device, desc="Val")
            train_metrics = evaluate(model, train_loader, loss_fn, device, desc="Train")
            
            history['eval_step'].append(step + 1)
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Evaluate symmetry loss on full datasets if penalty is active
            if symmetry_penalty is not None and lambda_sym > 0 and sym_layers:
                # Get full train data
                X_train = torch.cat([batch[0] for batch in train_loader], dim=0)
                X_val = torch.cat([batch[0] for batch in val_loader], dim=0)
                
                with torch.no_grad():
                    train_sym = symmetry_penalty.compute_total(
                        model, X_train, sym_layers, n_augmentations, device
                    ).item()
                    val_sym = symmetry_penalty.compute_total(
                        model, X_val, sym_layers, n_augmentations, device
                    ).item()
                
                history['train_sym_loss'].append(train_sym)
                history['val_sym_loss'].append(val_sym)
            else:
                history['train_sym_loss'].append(0.0)
                history['val_sym_loss'].append(0.0)
            
            # Save best model
            if save_best and save_dir is not None and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(model.state_dict(), save_dir / 'model_best.pt')
    
    pbar.close()
    
    # Save final model
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir / 'model.pt')
        
        # Save metrics (convert to serializable format)
        history_to_save = {k: v for k, v in history.items()}
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(history_to_save, f, indent=2)
    
    history['steps_per_epoch'] = steps_per_epoch
    return history


def plot_training_loss(
    ax: plt.Axes,
    history: dict,
    lambda_sym: float = 0.0,
    xlim: tuple = None,
    title: str = 'Training and Validation Loss',
):
    """
    Plot training loss curves on a given axes.
    
    This is the shared plotting function used by both plot_loss_curves and
    plot_run_summary to ensure consistent visualization.
    
    Plots task loss and (if present) symmetry loss on the same axes with
    different colors. Uses log scale for y-axis.
    
    Args:
        ax: Matplotlib axes to plot on.
        history: Training history dictionary with batch_step, batch_loss, etc.
        lambda_sym: Lambda weight for symmetry penalty (for scaling sym loss).
        xlim: Optional tuple (xmin, xmax) for fixed x-axis limits.
        title: Title for the axes.
    
    Returns:
        True if symmetry loss was plotted, False otherwise.
    """
    # Check for data using new or old history format
    eval_steps = history.get('eval_step', history.get('step', []))
    
    if not eval_steps:
        ax.text(0.5, 0.5, 'No training (steps=0)', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        return False
    
    eval_steps = np.array(eval_steps)
    train_loss_eval = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    
    # Get per-batch loss if available
    batch_steps = np.array(history.get('batch_step', []))
    batch_loss = np.array(history.get('batch_loss', []))
    
    # For backward compatibility: if no batch data, use eval data
    if len(batch_steps) == 0:
        batch_steps = eval_steps
        batch_loss = train_loss_eval
    
    # Get symmetry loss if available
    batch_sym_loss = np.array(history.get('batch_sym_loss', []))
    train_sym_loss = np.array(history.get('train_sym_loss', []))
    val_sym_loss = np.array(history.get('val_sym_loss', []))
    has_sym_loss = len(batch_sym_loss) > 0 and np.any(batch_sym_loss > 0) and lambda_sym > 0
    
    # Plot task loss
    ax.plot(batch_steps, batch_loss, color='tab:blue', alpha=0.3, linewidth=0.5, label='Task (batch)')
    ax.plot(eval_steps, train_loss_eval, color='tab:blue', linewidth=2, label='Task - Train')
    ax.plot(eval_steps, val_loss, color='tab:orange', linewidth=2, label='Task - Val')
    
    # Plot symmetry loss (scaled by lambda) if available
    if has_sym_loss:
        batch_sym_scaled = batch_sym_loss * lambda_sym
        train_sym_scaled = train_sym_loss * lambda_sym
        val_sym_scaled = val_sym_loss * lambda_sym
        ax.plot(batch_steps, batch_sym_scaled, color='tab:green', alpha=0.3, linewidth=0.5, label='λ·Sym (batch)')
        ax.plot(eval_steps, train_sym_scaled, color='tab:green', linewidth=2, label='λ·Sym - Train')
        ax.plot(eval_steps, val_sym_scaled, color='tab:red', linewidth=2, label='λ·Sym - Val')
    
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(fontsize='x-small', loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    return has_sym_loss


def plot_loss_curves(
    history,
    save_path,
    sym_penalty_name: str = None,
    sym_layers: List[int] = None,
    lambda_sym: float = 0.0,
    field_name: str = None,
):
    """
    Plot training loss curves.
    
    Creates a figure with:
    - Top/main: Combined task and symmetry loss on the same axes
    - Bottom (if gradient alignment data exists): Gradient cosine similarity
    
    Args:
        history: Training history dictionary.
        save_path: Path to save the plot.
        sym_penalty_name: Name of the symmetry penalty class.
        sym_layers: List of layer indices being penalized.
        lambda_sym: Lambda weight for symmetry penalty.
        field_name: Name of the scalar field used for training.
    """
    # Check if we have gradient alignment data
    grad_align_steps = history.get('grad_align_step', [])
    has_grad_align = len(grad_align_steps) > 0
    
    if has_grad_align:
        # Two rows: loss on top, gradient alignment on bottom
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
        ax_loss = axes[0]
        ax_grad = axes[1]
    else:
        # Single plot
        fig, ax_loss = plt.subplots(figsize=(10, 5))
    
    # Build suptitle with field name and training info
    suptitle_lines = []
    if field_name:
        suptitle_lines.append(f'Field = {field_name}')
    if sym_penalty_name and lambda_sym > 0 and sym_layers:
        layers_str = str(sym_layers).replace(' ', '')
        suptitle_lines.append(f'Model trained with {sym_penalty_name} penalty (layers={layers_str}, λ={lambda_sym})')
    elif field_name:
        suptitle_lines.append('Model trained without symmetry penalty')
    
    if suptitle_lines:
        fig.suptitle('\n'.join(suptitle_lines), fontsize=11, fontweight='bold')
    
    # Plot losses using shared function (no title on axes since we have suptitle)
    plot_training_loss(ax_loss, history, lambda_sym=lambda_sym, title='Training and Validation Loss')
    
    # Plot gradient alignment if available
    if has_grad_align:
        grad_align = np.array(history['grad_align'])
        grad_align_steps = np.array(grad_align_steps)
        
        ax_grad.plot(grad_align_steps, grad_align, color='purple', linewidth=1, alpha=0.7)
        ax_grad.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax_grad.set_xlabel('Step')
        ax_grad.set_ylabel('Cosine Similarity')
        ax_grad.set_title('Gradient Alignment (Task vs Symmetry)')
        ax_grad.set_ylim(-1.1, 1.1)
        ax_grad.grid(True, alpha=0.3)
        
        # Add colored regions to indicate alignment
        ax_grad.axhspan(-1, 0, alpha=0.1, color='red', label='Opposing')
        ax_grad.axhspan(0, 1, alpha=0.1, color='green', label='Aligned')
        ax_grad.legend(fontsize='x-small', loc='upper right')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
