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


def train(model, loss_fn, loader, optimizer):
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
        optimizer.step()
    
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
         num_epochs=10, run_seed=42, input_scale=None, input_scale_events=2000):
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
    model = DeepSets(
        in_channels=4,
        out_channels=num_kps,
        hidden_channels=hidden_channels,
        num_phi_layers=num_blocks,  # Use num_blocks for phi layers
        num_rho_layers=num_blocks,  # Use num_blocks for rho layers
        pool_mode='sum',  # Sum pooling preserves particle magnitudes
    ).to(device)
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    # Training loop
    pbar = tqdm(range(num_epochs))
    train_task_losses = []
    val_task_losses = []
    train_task_batch_losses = []
    train_rel_rmses = []
    val_rel_rmses = []
    
    for epoch in pbar:
        train_task_loss, task_batch_losses, train_rel_rmse = train(model, loss_fn, train_loader, optimizer)
        val_task_loss, val_rel_rmse = evaluate(model, loss_fn, val_loader)
        
        train_task_losses.append(train_task_loss)
        val_task_losses.append(val_task_loss)
        train_task_batch_losses.extend(task_batch_losses)
        train_rel_rmses.append(train_rel_rmse)
        val_rel_rmses.append(val_rel_rmse)
        
        pbar.set_postfix({
            'train_rel_rmse': f'{train_rel_rmse:.3f}',
            'val_rel_rmse': f'{val_rel_rmse:.3f}',
        })
    
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
    print(f'Model:             DeepSets')
    print(f'Learning rate:     {learning_rate:.2e}')
    print(f'Num layers:        {num_blocks}')
    print(f'Hidden channels:   {hidden_channels}')
    print(f'Batch size:        {batch_size}')
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
        'learning_rate': learning_rate,
        'num_blocks': num_blocks,
        'hidden_channels': hidden_channels,
        'num_heads': num_heads,
        'test_task_loss': test_task_loss,
        'test_relative_rmse': test_rel_rmse,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer model for kinematic polynomial prediction')
    parser.add_argument('--num-particles', type=int, default=128,
                        help='Max particles per event (default: 128)')
    parser.add_argument('--num-events', type=int, default=10_000,
                        help='Number of training events (default: 10_000)')
    parser.add_argument('--num-blocks', type=int, default=4,
                        help='Number of transformer blocks (default: 4)')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for optimizer (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--run-seed', type=int, default=42,
                        help='Base random seed for reproducibility (default: 42)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (skip plotting and visualization)')
    parser.add_argument('--input-scale', type=float, default=None,
                        help='Optional global scale to divide inputs by (default: auto-estimate)')
    parser.add_argument('--input-scale-events', type=int, default=2000,
                        help='Number of events to sample when auto-estimating input scale')
    
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
    )

