import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import hashlib
import hydra
from omegaconf import DictConfig, OmegaConf

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

def build_dataloader(n_events: int, n_particles: int, batch_size: int, input_scale: float, num_workers: int = 4, pin_memory: bool = True):
    """Build dataloader with fixed edges_list configuration."""
    edges_list = [[(0, 1), (0, 2), (0, 3)]]
    return make_kp_dataloader(
        edges_list=edges_list,
        n_events=n_events,
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    # Accumulate on GPU to avoid sync overhead (critical for MPS)
    total_loss = torch.tensor(0.0, device=device)
    sum_sq_error = torch.tensor(0.0, device=device)
    sum_y = torch.tensor(0.0, device=device)
    sum_y_sq = torch.tensor(0.0, device=device)
    count = 0
    num_batches = 0
    
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)  # (batch_size, num_particles, 4)
        yb = yb.to(device)  # (batch_size, 1, num_kps)
        
        # Squeeze the singleton dimension from yb: (batch_size, 1, num_kps) -> (batch_size, num_kps)
        yb = yb.squeeze(1)

        optimizer.zero_grad()
        pred = model(xb)  # (batch_size, num_kps)
        task_loss = loss_fn(pred, yb)
        
        # Accumulate on GPU (no .item() calls in the loop!)
        with torch.no_grad():
            total_loss += task_loss
            diff = pred - yb
            sum_sq_error += diff.pow(2).sum()
            sum_y += yb.sum()
            sum_y_sq += yb.pow(2).sum()
            count += yb.numel()
            num_batches += 1
        
        # Backprop
        task_loss.backward()
        
        # Gradient clipping
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Step scheduler per batch for warmup
        if scheduler is not None:
            scheduler.step()
    
    # Only sync with CPU at end of epoch
    avg_task_loss = (total_loss / num_batches).item()
    _, rel_rmse = _accumulate_regression_stats(
        sum_sq_error.item(), sum_y.item(), sum_y_sq.item(), count
    )
    
    return avg_task_loss, [], rel_rmse  # Empty list for batch losses (not tracked anymore)

def evaluate(model, loss_fn, loader):
    """Evaluation loop."""
    model.eval()
    # Accumulate on GPU to avoid sync overhead (critical for MPS)
    total_loss = torch.tensor(0.0, device=device)
    sum_sq_error = torch.tensor(0.0, device=device)
    sum_y = torch.tensor(0.0, device=device)
    sum_y_sq = torch.tensor(0.0, device=device)
    count = 0
    num_batches = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)  # (batch_size, num_particles, 4)
            yb = yb.to(device)  # (batch_size, 1, num_kps)
            
            # Squeeze the singleton dimension from yb
            yb = yb.squeeze(1)

            pred = model(xb)  # (batch_size, num_kps)
            total_loss += loss_fn(pred, yb)
            diff = pred - yb
            sum_sq_error += diff.pow(2).sum()
            sum_y += yb.sum()
            sum_y_sq += yb.pow(2).sum()
            count += yb.numel()
            num_batches += 1
    
    # Only sync with CPU at end
    task_loss = (total_loss / num_batches).item()
    _, rel_rmse = _accumulate_regression_stats(
        sum_sq_error.item(), sum_y.item(), sum_y_sq.item(), count
    )
    return task_loss, rel_rmse

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    
    Returns:
        Dictionary with training results
    """
    # Print configuration
    print('\n' + '='*60)
    print('CONFIGURATION')
    print('='*60)
    print(OmegaConf.to_yaml(cfg))
    print('='*60 + '\n')
    
    # Extract config values
    headless = cfg.headless
    learning_rate = cfg.training.learning_rate
    hidden_channels = cfg.model.hidden_channels
    num_events = cfg.data.num_events
    n_particles = cfg.data.n_particles
    batch_size = cfg.data.batch_size
    num_epochs = cfg.training.num_epochs
    run_seed = cfg.run_seed
    input_scale = cfg.data.input_scale
    input_scale_events = cfg.data.input_scale_events
    model_type = cfg.model.type
    warmup_epochs = cfg.training.warmup_epochs
    weight_decay = cfg.training.weight_decay
    grad_clip = cfg.training.grad_clip
    dropout = cfg.training.dropout
    
    # Model-specific parameters
    if model_type == 'deepsets':
        num_phi_layers = cfg.model.num_phi_layers
        num_rho_layers = cfg.model.num_rho_layers
        num_blocks = None  # Not used for DeepSets
        num_heads = None  # Not used for DeepSets
    else:  # transformer
        num_blocks = cfg.model.num_blocks
        num_heads = cfg.model.num_heads
        num_phi_layers = None  # Not used for Transformer
        num_rho_layers = None  # Not used for Transformer
    
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
    num_workers = cfg.data.get('num_workers', 4)
    pin_memory = cfg.data.get('pin_memory', True)
    
    train_loader = build_dataloader(
        n_events=int(num_events * cfg.data.train_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        n_events=int(num_events * cfg.data.val_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_dataloader(
        n_events=int(num_events * cfg.data.test_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
            num_phi_layers=num_phi_layers,
            num_rho_layers=num_rho_layers,
            pool_mode=cfg.model.get('pool_mode', 'sum'),
        ).to(device)
    elif model_type == 'transformer':
        dropout_prob = dropout if dropout > 0 else None
        if dropout_prob is None:
            dropout_prob = cfg.model.get('dropout_prob')
        model = Transformer(
            in_channels=4,
            out_channels=num_kps,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            use_mean_pooling=cfg.model.get('use_mean_pooling', True),
            dropout_prob=dropout_prob,
            multi_query=cfg.model.get('multi_query', False),
            increase_hidden_channels=cfg.model.get('increase_hidden_channels', 1),
            checkpoint_blocks=cfg.model.get('checkpoint_blocks', False),
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
    print(f'Number of training events: {int(num_events * cfg.data.train_split)}')
    print(f'Number of validation events: {int(num_events * cfg.data.val_split)}')
    print(f'Number of test events: {int(num_events * cfg.data.test_split)}')
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
    if model_type == 'deepsets':
        print(f'Num phi layers:    {num_phi_layers}')
        print(f'Num rho layers:    {num_rho_layers}')
    else:  # transformer
        print(f'Num blocks:        {num_blocks}')
        print(f'Num heads:         {num_heads}')
    print(f'Hidden channels:   {hidden_channels}')
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
    result = {
        'model_type': model_type,
        'learning_rate': learning_rate,
        'hidden_channels': hidden_channels,
        'test_task_loss': test_task_loss,
        'test_relative_rmse': test_rel_rmse,
        'best_val_relative_rmse': best_val_rmse,
    }
    if model_type == 'deepsets':
        result['num_phi_layers'] = num_phi_layers
        result['num_rho_layers'] = num_rho_layers
    else:  # transformer
        result['num_blocks'] = num_blocks
        result['num_heads'] = num_heads
    return result


if __name__ == '__main__':
    main()

