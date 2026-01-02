import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import hashlib
import argparse

from models import Transformer, DeepSets
from data.kp_dataset import make_kp_dataloader, estimate_input_scale

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """
    Create a schedule with linear warmup and cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def derive_seed(run_seed: int, category: str) -> int:
    """
    Deterministically derive a seed from a base seed and category name.
    
    Args:
        run_seed: Base seed for the experiment run
        category: Category name (e.g., "data", "model", "augmentation")
    
    Returns:
        Derived seed value
    """
    # Use hashlib for deterministic derivation across Python sessions
    # Python's built-in hash() is non-deterministic due to hash randomization
    seed_str = f"{run_seed}_{category}"
    seed_bytes = seed_str.encode('utf-8')
    hash_obj = hashlib.md5(seed_bytes)
    hash_int = int(hash_obj.hexdigest(), 16)

    return hash_int % (2**31)

def set_model_seed(seed):
    """Set random seeds for model initialization (weights, dropout, etc.)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataloader(n_events: int, n_particles: int, batch_size: int, input_scale: float):
    """Build dataloader with fixed edges_list configuration."""
    edges_list = [[(0, 1), (0, 2), (0, 3)]]
    return make_kp_dataloader(
        edges_list=edges_list,
        n_events=n_events,
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
    )


def _accumulate_regression_stats(sum_sq_error: float, sum_y: float, sum_y_sq: float, count: int, eps: float = 1e-12):
    """Compute MSE and relative RMSE from accumulated sums."""
    if count == 0:
        return float("nan"), float("nan")
    mse = sum_sq_error / count
    variance = (sum_y_sq / count) - (sum_y / count) ** 2
    std = math.sqrt(max(variance, 0.0))
    rel_rmse = math.sqrt(mse) / max(std, eps)
    return mse, rel_rmse


def train(model, loss_fn, loader, optimizer, scheduler=None, grad_clip=None):
    """Training loop."""
    task_batch_losses = []
    sum_sq_error = 0.0
    sum_y = 0.0
    sum_y_sq = 0.0
    count = 0
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)  # (batch_size, num_particles, 4)
        yb = yb.to(device)  # (batch_size, 1, num_kps)
        
        # Squeeze the singleton dimension from yb: (batch_size, 1, num_kps) -> (batch_size, num_kps)
        yb = yb.squeeze(1)

        optimizer.zero_grad()
        # Transformer with mean pooling returns (batch_size, num_kps) directly
        pred = model(xb)  # (batch_size, num_kps)
        task_loss = loss_fn(pred, yb)
        
        # Track losses
        task_batch_losses.append(task_loss.item())
        diff = (pred - yb).detach()
        sum_sq_error += diff.pow(2).sum().item()
        sum_y += yb.detach().sum().item()
        sum_y_sq += yb.detach().pow(2).sum().item()
        count += yb.numel()
        
        # Backprop
        task_loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Step scheduler per batch for warmup
        if scheduler is not None:
            scheduler.step()
    
    avg_task_loss = sum(task_batch_losses) / len(task_batch_losses)
    _, rel_rmse = _accumulate_regression_stats(sum_sq_error, sum_y, sum_y_sq, count)
    
    return avg_task_loss, task_batch_losses, rel_rmse

def evaluate(model, loss_fn, loader):
    """Evaluation loop."""
    model.eval()
    with torch.no_grad():
        task_loss = 0
        sum_sq_error = 0.0
        sum_y = 0.0
        sum_y_sq = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)  # (batch_size, num_particles, 4)
            yb = yb.to(device)  # (batch_size, 1, num_kps)
            
            # Squeeze the singleton dimension from yb
            yb = yb.squeeze(1)

            pred = model(xb)  # (batch_size, num_kps)
            task_loss += loss_fn(pred, yb).item()
            diff = pred - yb
            sum_sq_error += diff.pow(2).sum().item()
            sum_y += yb.sum().item()
            sum_y_sq += yb.pow(2).sum().item()
            count += yb.numel()
        task_loss /= len(loader)
        _, rel_rmse = _accumulate_regression_stats(sum_sq_error, sum_y, sum_y_sq, count)
        return task_loss, rel_rmse

def main(headless=False, learning_rate=3e-4, num_blocks=4, hidden_channels=128, 
         num_heads=4, num_events=10_000, n_particles=128, batch_size=256, 
         num_epochs=10, run_seed=42, input_scale=None, input_scale_events=2000,
         model_type='deepsets', warmup_epochs=5, weight_decay=0.01, grad_clip=1.0,
         dropout=0.0):
    """
    Main training function.
    
    Args:
        headless: If True, skip plotting
        learning_rate: Learning rate for optimizer
        num_blocks: Number of transformer blocks
        hidden_channels: Size of hidden channels
        num_heads: Number of attention heads
        num_events: Number of training events
        n_particles: Max particles per event
        batch_size: Batch size
        num_epochs: Number of training epochs
        run_seed: Base random seed for reproducibility
        model_type: 'deepsets' or 'transformer'
        warmup_epochs: Number of warmup epochs for LR scheduler
        weight_decay: Weight decay for AdamW
        grad_clip: Gradient clipping max norm
        dropout: Dropout probability
    
    Returns:
        Dictionary with training results
    """
    # Derive seeds for different randomness sources
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    
    # Set model seed before model creation
    set_model_seed(model_seed)
    
    # Set data generation seed
    np.random.seed(data_seed)
    random.seed(data_seed)
    
    # Estimate global input scale once, reuse across splits
    if input_scale is None:
        input_scale = estimate_input_scale(
            n_events=input_scale_events,
            n_particles=n_particles,
            seed=data_seed,
        )
        # Reseed to keep dataset generation deterministic
        np.random.seed(data_seed)
        random.seed(data_seed)
    # Build dataloaders
    train_loader = build_dataloader(
        n_events=int(num_events * 0.6),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
    )
    val_loader = build_dataloader(
        n_events=int(num_events * 0.2),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
    )
    test_loader = build_dataloader(
        n_events=int(num_events * 0.2),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
    )
    
    # Create model
    # Input: 4 channels (E, px, py, pz)
    # Output: 1 KP value (from edges_list = [[(0, 1), (0, 2), (0, 3)]])
    num_kps = 1
    
    if model_type == 'deepsets':
        model = DeepSets(
            in_channels=4,
            out_channels=num_kps,
            hidden_channels=hidden_channels,
            num_phi_layers=num_blocks,
            num_rho_layers=num_blocks,
            pool_mode='sum',
        ).to(device)
    elif model_type == 'transformer':
        model = Transformer(
            in_channels=4,
            out_channels=num_kps,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            use_mean_pooling=True,
            dropout_prob=dropout if dropout > 0 else None,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Calculate total training steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    pbar = tqdm(range(num_epochs))
    train_task_losses = []
    val_task_losses = []
    train_task_batch_losses = []
    train_rel_rmses = []
    val_rel_rmses = []
    best_val_rmse = float('inf')
    best_model_state = None
    
    for epoch in pbar:
        train_task_loss, task_batch_losses, train_rel_rmse = train(
            model, loss_fn, train_loader, optimizer, scheduler, grad_clip
        )
        val_task_loss, val_rel_rmse = evaluate(model, loss_fn, val_loader)
        
        train_task_losses.append(train_task_loss)
        val_task_losses.append(val_task_loss)
        train_task_batch_losses.extend(task_batch_losses)
        train_rel_rmses.append(train_rel_rmse)
        val_rel_rmses.append(val_rel_rmse)
        
        # Track best model
        if val_rel_rmse < best_val_rmse:
            best_val_rmse = val_rel_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        pbar.set_postfix({
            'train_rel_rmse': f'{train_rel_rmse:.3f}',
            'val_rel_rmse': f'{val_rel_rmse:.3f}',
            'best_val': f'{best_val_rmse:.3f}',
        })
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    test_task_loss, test_rel_rmse = evaluate(model, loss_fn, test_loader)
    
    # Print data section
    print('\n' + '='*25)
    print('DATA')
    print('='*25)
    print(f'Number of training events: {int(num_events * 0.6)}')
    print(f'Number of validation events: {int(num_events * 0.2)}')
    print(f'Number of test events: {int(num_events * 0.2)}')
    print(f'Max particles per event: {n_particles}')
    print(f'Number of epochs: {len(train_task_losses)}')
    print(f'Kinematic polynomial: edges_list = [[(0, 1), (0, 2), (0, 3)]]')
    print('='*25)
    
    # Print configuration section
    print('\n' + '='*25)
    print('CONFIGURATION')
    print('='*25)
    print(f'Model:             {model_type.capitalize()}')
    print(f'Learning rate:     {learning_rate:.2e}')
    print(f'Num layers:        {num_blocks}')
    print(f'Hidden channels:   {hidden_channels}')
    if model_type == 'transformer':
        print(f'Num heads:         {num_heads}')
    print(f'Batch size:        {batch_size}')
    print(f'Weight decay:      {weight_decay}')
    print(f'Warmup epochs:     {warmup_epochs}')
    print(f'Grad clip:         {grad_clip}')
    print(f'Input scale:       {input_scale:.3e}')
    print('='*25)
    
    # Print results section
    print('\n' + '='*25)
    print('RESULTS')
    print('='*25)
    print(f'Test task loss:     {test_task_loss:.4e}')
    print(f'Test relative RMSE: {test_rel_rmse:.4f}')
    print('='*25)
    
    # Only plot if not in headless mode
    if not headless:
        # Simple loss plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot batch losses behind epoch losses
        if train_task_batch_losses:
            n_batches = len(train_task_batch_losses)
            n_epochs = len(train_task_losses)
            batch_x = np.linspace(0, n_epochs - 1, n_batches)
            ax.plot(batch_x, train_task_batch_losses, alpha=0.3, color='k', linewidth=0.5, label='train (batch)')
        
        # Plot epoch losses
        epochs = range(len(train_task_losses))
        ax.plot(epochs, train_task_losses, label='train (epoch)', linewidth=1, color='k')
        ax.plot(epochs, val_task_losses, label='val', linewidth=1, color='b')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Task Loss', color='k')
        ax.set_yscale('log')
        ax.set_title('Task Loss')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.show()
    
    # Return results dictionary
    return {
        'model_type': model_type,
        'learning_rate': learning_rate,
        'num_blocks': num_blocks,
        'hidden_channels': hidden_channels,
        'num_heads': num_heads,
        'test_task_loss': test_task_loss,
        'test_relative_rmse': test_rel_rmse,
        'best_val_relative_rmse': best_val_rmse,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for kinematic polynomial prediction')
    parser.add_argument('--num-particles', type=int, default=128,
                        help='Max particles per event (default: 128)')
    parser.add_argument('--num-events', type=int, default=50_000,
                        help='Number of training events (default: 50_000)')
    parser.add_argument('--num-blocks', type=int, default=6,
                        help='Number of transformer blocks (default: 6)')
    parser.add_argument('--hidden-channels', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for optimizer (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--run-seed', type=int, default=42,
                        help='Base random seed for reproducibility (default: 42)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (skip plotting and visualization)')
    parser.add_argument('--input-scale', type=float, default=None,
                        help='Optional global scale to divide inputs by (default: auto-estimate)')
    parser.add_argument('--input-scale-events', type=int, default=2000,
                        help='Number of events to sample when auto-estimating input scale')
    parser.add_argument('--model-type', type=str, default='transformer',
                        choices=['deepsets', 'transformer'],
                        help='Model architecture (default: transformer)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs for LR scheduler (default: 5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for AdamW (default: 0.01)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability (default: 0.0)')
    
    args = parser.parse_args()
    
    main(
        headless=args.headless,
        learning_rate=args.learning_rate,
        num_blocks=args.num_blocks,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_events=args.num_events,
        n_particles=args.num_particles,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        run_seed=args.run_seed,
        input_scale=args.input_scale,
        input_scale_events=args.input_scale_events,
        model_type=args.model_type,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
    )

