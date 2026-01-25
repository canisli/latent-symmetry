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
    frame_callback: Optional[Callable[[int, nn.Module, Dict], None]] = None,
    frame_interval: int = 10,
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
    
    Returns:
        Dictionary of training metrics history.
    """
    model.to(device)
    
    history = {
        'batch_step': [],         # Step number for each batch
        'batch_loss': [],         # Per-batch task loss (training dynamics)
        'batch_sym_loss': [],     # Per-batch symmetry loss
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
        
        # Training step
        task_loss, sym_loss = train_step(
            model, batch, loss_fn, optimizer, device,
            symmetry_penalty=symmetry_penalty,
            lambda_sym=lambda_sym,
            sym_layers=sym_layers,
            n_augmentations=n_augmentations,
        )
        
        # Record per-batch losses
        history['batch_step'].append(step + 1)
        history['batch_loss'].append(task_loss)
        history['batch_sym_loss'].append(sym_loss)
        
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


def plot_loss_curves(
    history,
    save_path,
    sym_penalty_name: str = None,
    sym_layers: List[int] = None,
    lambda_sym: float = 0.0,
):
    """
    Plot training loss curves.
    
    Shows two types of training loss:
    - Batch loss: Per-batch loss at each training step (training dynamics)
    - Eval loss: Full-dataset evaluation (true performance)
    
    If symmetry loss is present and non-zero, creates two subplots:
    - Left: Task loss (batch for train dynamics, eval for train/val performance)
    - Right: Symmetry loss (batch for train dynamics, eval for train/val performance)
    
    Otherwise, creates a single plot with task loss.
    
    Args:
        history: Training history dictionary.
        save_path: Path to save the plot.
        sym_penalty_name: Name of the symmetry penalty class (e.g., "RawOrbitVariancePenalty").
        sym_layers: List of layer indices being penalized.
        lambda_sym: Lambda weight for symmetry penalty.
    """
    # Per-batch data (recorded every step)
    batch_steps = np.array(history.get('batch_step', []))
    batch_loss = np.array(history.get('batch_loss', []))
    batch_sym_loss = np.array(history.get('batch_sym_loss', []))
    
    # Evaluation data (recorded at eval intervals)
    eval_steps = np.array(history.get('eval_step', history.get('step', [])))
    train_loss_eval = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    train_sym_loss = np.array(history.get('train_sym_loss', []))
    val_sym_loss = np.array(history.get('val_sym_loss', []))
    
    # Check if we have non-trivial symmetry loss
    has_sym_loss = len(batch_sym_loss) > 0 and np.any(batch_sym_loss > 0)
    
    # For backward compatibility: if no batch data, use eval data for both
    if len(batch_steps) == 0:
        batch_steps = eval_steps
        batch_loss = train_loss_eval

    if has_sym_loss:
        # Two subplots: task loss and symmetry loss
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Task loss
        ax = axes[0]
        ax.plot(batch_steps, batch_loss, color='tab:blue', alpha=0.3, linewidth=0.5, label='Train (batch)')
        ax.plot(eval_steps, train_loss_eval, color='tab:blue', linewidth=2, label='Train (eval)')
        ax.plot(eval_steps, val_loss, color='tab:orange', linewidth=2, label='Val (eval)')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Task Loss (MSE)')
        ax.set_title('Task Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right: Symmetry loss
        ax = axes[1]
        ax.plot(batch_steps, batch_sym_loss, color='tab:blue', alpha=0.3, linewidth=0.5, label='Train (batch)')
        ax.plot(eval_steps, train_sym_loss, color='tab:blue', linewidth=2, label='Train (eval)')
        ax.plot(eval_steps, val_sym_loss, color='tab:orange', linewidth=2, label='Val (eval)')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Orbit Variance')
        
        # Build symmetry loss title with penalty info
        sym_title = sym_penalty_name if sym_penalty_name else 'Symmetry Loss'
        sym_subtitle_parts = []
        if sym_layers:
            layers_str = ', '.join(str(l) for l in sym_layers)
            sym_subtitle_parts.append(f'layers=[{layers_str}]')
        if lambda_sym > 0:
            sym_subtitle_parts.append(f'λ={lambda_sym}')
        if sym_subtitle_parts:
            sym_title += f'\n({", ".join(sym_subtitle_parts)})'
        ax.set_title(sym_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.suptitle('Training Loss Curves', fontsize=14)
        plt.tight_layout()
    else:
        # Single plot: task loss only
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(batch_steps, batch_loss, color='tab:blue', alpha=0.3, linewidth=0.5, label='Train (batch)')
        ax.plot(eval_steps, train_loss_eval, color='tab:blue', linewidth=2, label='Train (eval)')
        ax.plot(eval_steps, val_loss, color='tab:orange', linewidth=2, label='Val (eval)')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
