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
) -> Tuple[float, float]:
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
    
    Returns:
        Tuple of (task_loss, sym_loss) as floats.
    """
    model.train()
    X, y = batch
    X, y = X.to(device), y.to(device)
    
    optimizer.zero_grad()
    preds = model(X)
    task_loss = loss_fn(preds, y)
    
    sym_loss_value = 0.0
    if symmetry_penalty is not None and lambda_sym > 0 and sym_layers:
        sym_loss = symmetry_penalty.compute_total(
            model, X, sym_layers, n_augmentations, device
        )
        total_loss = task_loss + lambda_sym * sym_loss
        sym_loss_value = sym_loss.item()
    else:
        total_loss = task_loss
    
    total_loss.backward()
    optimizer.step()
    
    return task_loss.item(), sym_loss_value


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
    
    Returns:
        Dictionary of training metrics history.
    """
    model.to(device)
    
    history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'lr': [],
        'sym_loss': [],
    }
    steps_per_epoch = len(train_loader)
    
    best_val_loss = float('inf')
    step = 0
    train_iter = iter(train_loader)
    running_task_loss = 0.0
    running_sym_loss = 0.0
    running_count = 0
    
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:
        # Get next batch (cycle through dataloader)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Training step
        task_loss, sym_loss = train_step(
            model, batch, loss_fn, optimizer, device,
            symmetry_penalty=symmetry_penalty,
            lambda_sym=lambda_sym,
            sym_layers=sym_layers,
            n_augmentations=n_augmentations,
        )
        running_task_loss += task_loss
        running_sym_loss += sym_loss
        running_count += 1
        
        if scheduler is not None:
            scheduler.step()
        
        # Evaluation
        if step % eval_interval == 0 or step == total_steps - 1:
            val_metrics = evaluate(model, val_loader, loss_fn, device, desc="Val")
            train_metrics = evaluate(model, train_loader, loss_fn, device, desc="Train")
            
            history['step'].append(step + 1)
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['lr'].append(optimizer.param_groups[0]['lr'])
            # Record average symmetry loss since last eval
            avg_sym_loss = running_sym_loss / running_count if running_count > 0 else 0.0
            history['sym_loss'].append(avg_sym_loss)
            
            # Reset running averages
            running_task_loss = 0.0
            running_sym_loss = 0.0
            running_count = 0
            
            # Save best model
            if save_best and save_dir is not None and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(model.state_dict(), save_dir / 'model_best.pt')
    
    pbar.close()
    
    # Save final model
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir / 'model.pt')
        
        # Save metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(history, f, indent=2)
    history['steps_per_epoch'] = steps_per_epoch
    return history


def plot_loss_curves(history, save_path):
    """
    Plot training loss curves.
    
    If symmetry loss is present and non-zero, creates two subplots:
    - Left: Task loss (train/val)
    - Right: Symmetry loss
    
    Otherwise, creates a single plot with task loss.
    """
    steps = np.array(history['step'])
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    steps_per_epoch = max(1, int(history.get('steps_per_epoch', 1)))
    
    # Check if we have non-trivial symmetry loss
    sym_loss = np.array(history.get('sym_loss', []))
    has_sym_loss = len(sym_loss) > 0 and np.any(sym_loss > 0)

    epoch_idx = (steps - 1) // steps_per_epoch
    epoch_vals = np.unique(epoch_idx)
    epoch_train = np.array([train_loss[epoch_idx == e].mean() for e in epoch_vals])
    epoch_val = np.array([val_loss[epoch_idx == e].mean() for e in epoch_vals])

    if has_sym_loss:
        # Two subplots: task loss and symmetry loss
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Task loss
        ax = axes[0]
        ax.plot(steps, train_loss, color='tab:blue', alpha=0.25, linewidth=1, label='Train (step)')
        ax.plot(steps, val_loss, color='tab:orange', alpha=0.25, linewidth=1, label='Val (step)')
        ax.plot((epoch_vals + 1) * steps_per_epoch, epoch_train, color='tab:blue', linewidth=2, label='Train (epoch)')
        ax.plot((epoch_vals + 1) * steps_per_epoch, epoch_val, color='tab:orange', linewidth=2, label='Val (epoch)')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Task Loss (MSE)')
        ax.set_title('Task Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right: Symmetry loss
        ax = axes[1]
        epoch_sym = np.array([sym_loss[epoch_idx == e].mean() for e in epoch_vals])
        ax.plot(steps, sym_loss, color='tab:green', alpha=0.25, linewidth=1, label='Sym Loss (step)')
        ax.plot((epoch_vals + 1) * steps_per_epoch, epoch_sym, color='tab:green', linewidth=2, label='Sym Loss (epoch)')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Symmetry Loss (Orbit Variance)')
        ax.set_title('Symmetry Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.suptitle('Training Loss Curves', fontsize=14)
        plt.tight_layout()
    else:
        # Single plot: task loss only
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, train_loss, color='tab:blue', alpha=0.25, linewidth=1, label='Train (step)')
        ax.plot(steps, val_loss, color='tab:orange', alpha=0.25, linewidth=1, label='Val (step)')
        ax.plot((epoch_vals + 1) * steps_per_epoch, epoch_train, color='tab:blue', linewidth=2, label='Train (epoch)')
        ax.plot((epoch_vals + 1) * steps_per_epoch, epoch_val, color='tab:orange', linewidth=2, label='Val (epoch)')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
