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
from data.kp_dataset import make_kp_dataloader
from symmetry import lorentz_orbit_variance_loss

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


def train(
    model,
    loss_fn,
    loader,
    optimizer,
    scheduler=None,
    grad_clip=None,
    symmetry_layer=None,
    lambda_sym=0.0,
    std_eta=0.5,
    augmentation_generator=None,
):
    """Training loop with optional symmetry loss.
    
    Args:
        model: The model to train
        loss_fn: Task loss function
        loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        grad_clip: Optional gradient clipping value
        symmetry_layer: Layer index for symmetry loss (None to disable)
        lambda_sym: Weight for symmetry loss
        std_eta: Rapidity std for Lorentz augmentations
        augmentation_generator: Optional random generator for reproducibility
    
    Returns:
        Tuple of (avg_task_loss, task_batch_losses, rel_rmse, avg_sym_loss, sym_batch_losses)
    """
    task_batch_losses = []
    sym_batch_losses = []
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
        pred = model(xb)  # (batch_size, num_kps)
        task_loss = loss_fn(pred, yb)
        
        # Compute symmetry loss if enabled
        sym_loss = torch.tensor(0.0, device=device)
        if symmetry_layer is not None and lambda_sym > 0:
            sym_loss = lorentz_orbit_variance_loss(
                model, xb, symmetry_layer,
                std_eta=std_eta,
                generator=augmentation_generator,
            )
        
        # Track losses separately
        task_batch_losses.append(task_loss.item())
        sym_batch_losses.append(sym_loss.item())
        diff = (pred - yb).detach()
        sum_sq_error += diff.pow(2).sum().item()
        sum_y += yb.detach().sum().item()
        sum_y_sq += yb.detach().pow(2).sum().item()
        count += yb.numel()
        
        # Total loss for backprop
        total_loss = task_loss + lambda_sym * sym_loss
        total_loss.backward()
        
        # Gradient clipping
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Step scheduler per batch for warmup
        if scheduler is not None:
            scheduler.step()
    
    avg_task_loss = sum(task_batch_losses) / len(task_batch_losses)
    avg_sym_loss = sum(sym_batch_losses) / len(sym_batch_losses) if sym_batch_losses else 0.0
    _, rel_rmse = _accumulate_regression_stats(sum_sq_error, sum_y, sum_y_sq, count)
    
    return avg_task_loss, task_batch_losses, rel_rmse, avg_sym_loss, sym_batch_losses

def evaluate(
    model,
    loss_fn,
    loader,
    symmetry_layer=None,
    std_eta=0.5,
    augmentation_generator=None,
):
    """Evaluation loop with optional symmetry loss measurement.
    
    Args:
        model: The model to evaluate
        loss_fn: Task loss function
        loader: DataLoader for evaluation data
        symmetry_layer: Layer index for symmetry loss (None to disable)
        std_eta: Rapidity std for Lorentz augmentations
        augmentation_generator: Optional random generator for reproducibility
    
    Returns:
        Tuple of (task_loss, rel_rmse, sym_loss)
    """
    model.eval()
    with torch.no_grad():
        task_loss = 0
        sym_loss = 0
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
            
            if symmetry_layer is not None:
                sym_loss += lorentz_orbit_variance_loss(
                    model, xb, symmetry_layer,
                    std_eta=std_eta,
                    generator=augmentation_generator,
                ).item()
            
            diff = pred - yb
            sum_sq_error += diff.pow(2).sum().item()
            sum_y += yb.sum().item()
            sum_y_sq += yb.pow(2).sum().item()
            count += yb.numel()
        
        task_loss /= len(loader)
        if symmetry_layer is not None:
            sym_loss /= len(loader)
        _, rel_rmse = _accumulate_regression_stats(sum_sq_error, sum_y, sum_y_sq, count)
        return task_loss, rel_rmse, sym_loss

def run_training(
    # Data params
    num_events: int = 10000,
    n_particles: int = 128,
    batch_size: int = 256,
    input_scale: float = 0.9515689,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    # Training params
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    warmup_epochs: int = 5,
    weight_decay: float = 0.0,
    grad_clip: float = None,
    dropout: float = 0.0,
    early_stopping_patience: int = None,
    # Model params
    model_type: str = 'deepsets',
    hidden_channels: int = 128,
    num_phi_layers: int = None,
    num_rho_layers: int = None,
    pool_mode: str = 'sum',
    num_blocks: int = None,
    num_heads: int = None,
    use_mean_pooling: bool = True,
    dropout_prob: float = None,
    multi_query: bool = False,
    increase_hidden_channels: int = 1,
    checkpoint_blocks: bool = False,
    # Symmetry params
    symmetry_enabled: bool = False,
    symmetry_layer: int = None,
    lambda_sym_max: float = 1.0,
    std_eta: float = 0.5,
    # Other
    run_seed: int = 42,
    headless: bool = False,
):
    """
    Core training function that can be called directly.
    
    Returns:
        Dictionary with training results including:
        - test_task_loss, test_relative_rmse, test_sym_loss (if symmetry enabled)
        - model and training configuration
    """
    # Derive seeds for different randomness sources
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    augmentation_seed = derive_seed(run_seed, "augmentation")
    
    # Set model seed before model creation
    set_model_seed(model_seed)
    
    # Set data generation seed
    np.random.seed(data_seed)
    random.seed(data_seed)
    
    # Build dataloaders 
    train_loader = build_dataloader(
        n_events=int(num_events * train_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
    )
    val_loader = build_dataloader(
        n_events=int(num_events * val_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
    )
    test_loader = build_dataloader(
        n_events=int(num_events * test_split),
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
            num_phi_layers=num_phi_layers,
            num_rho_layers=num_rho_layers,
            pool_mode=pool_mode,
        ).to(device)
    elif model_type == 'transformer':
        _dropout_prob = dropout if dropout > 0 else dropout_prob
        model = Transformer(
            in_channels=4,
            out_channels=num_kps,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            use_mean_pooling=use_mean_pooling,
            dropout_prob=_dropout_prob,
            multi_query=multi_query,
            increase_hidden_channels=increase_hidden_channels,
            checkpoint_blocks=checkpoint_blocks,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create augmentation generator for symmetry loss
    augmentation_generator = torch.Generator(device=device).manual_seed(augmentation_seed)
    
    # Calculate total training steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    pbar = tqdm(range(num_epochs), disable=headless)
    train_task_losses = []
    val_task_losses = []
    train_task_batch_losses = []
    train_sym_batch_losses = []
    train_rel_rmses = []
    val_rel_rmses = []
    train_sym_losses = []
    val_sym_losses = []
    best_val_loss = float('inf')
    best_val_rmse = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    # Constant lambda for symmetry loss
    lambda_sym = lambda_sym_max if symmetry_enabled else 0.0
    
    for epoch in pbar:
        train_task_loss, task_batch_losses, train_rel_rmse, train_sym_loss, sym_batch_losses = train(
            model, loss_fn, train_loader, optimizer, scheduler, grad_clip,
            symmetry_layer=symmetry_layer,
            lambda_sym=lambda_sym,
            std_eta=std_eta,
            augmentation_generator=augmentation_generator,
        )
        val_task_loss, val_rel_rmse, val_sym_loss = evaluate(
            model, loss_fn, val_loader,
            symmetry_layer=symmetry_layer,
            std_eta=std_eta,
            augmentation_generator=augmentation_generator,
        )
        
        train_task_losses.append(train_task_loss)
        val_task_losses.append(val_task_loss)
        train_task_batch_losses.extend(task_batch_losses)
        train_sym_batch_losses.extend(sym_batch_losses)
        train_rel_rmses.append(train_rel_rmse)
        val_rel_rmses.append(val_rel_rmse)
        train_sym_losses.append(train_sym_loss)
        val_sym_losses.append(val_sym_loss)
        
        # Track best model (by validation task loss for early stopping)
        if val_task_loss < best_val_loss:
            best_val_loss = val_task_loss
            best_val_rmse = val_rel_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        postfix = {
            'train_rel_rmse': f'{train_rel_rmse:.3f}',
            'val_rel_rmse': f'{val_rel_rmse:.3f}',
            'best_val': f'{best_val_rmse:.3f}',
        }
        if symmetry_enabled:
            postfix['sym'] = f'{train_sym_loss:.2e}'
            postfix['λ'] = f'{lambda_sym:.3f}'
        pbar.set_postfix(postfix)
        
        # Early stopping check
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
            break
    
    # Track how many epochs actually ran
    epochs_trained = len(train_task_losses)
    early_stopped = epochs_trained < num_epochs
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    test_task_loss, test_rel_rmse, test_sym_loss = evaluate(
        model, loss_fn, test_loader,
        symmetry_layer=symmetry_layer,
        std_eta=std_eta,
        augmentation_generator=augmentation_generator,
    )
    
    if not headless:
        # Print data section
        print('\n' + '='*25)
        print('DATA')
        print('='*25)
        print(f'Number of training events: {int(num_events * train_split)}')
        print(f'Number of validation events: {int(num_events * val_split)}')
        print(f'Number of test events: {int(num_events * test_split)}')
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
        if symmetry_enabled:
            print(f'Symmetry layer:    {symmetry_layer}')
            print(f'Lambda sym max:    {lambda_sym_max}')
            print(f'Std eta:           {std_eta}')
        print('='*25)
        
        # Print results section
        print('\n' + '='*25)
        print('RESULTS')
        print('='*25)
        print(f'Test task loss:     {test_task_loss:.4e}')
        print(f'Test relative RMSE: {test_rel_rmse:.4f}')
        if symmetry_enabled:
            print(f'Test symmetry loss: {test_sym_loss:.4e}')
        print('='*25)
        
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
        'epochs_trained': epochs_trained,
        'num_epochs': num_epochs,
        'early_stopped': early_stopped,
    }
    if model_type == 'deepsets':
        result['num_phi_layers'] = num_phi_layers
        result['num_rho_layers'] = num_rho_layers
    else:  # transformer
        result['num_blocks'] = num_blocks
        result['num_heads'] = num_heads
    if symmetry_enabled:
        result['symmetry_layer'] = symmetry_layer
        result['lambda_sym_max'] = lambda_sym_max
        result['std_eta'] = std_eta
        result['test_sym_loss'] = test_sym_loss
    return result


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point using Hydra configuration.
    Extracts config values and calls run_training().
    """
    # Print configuration
    print('\n' + '='*60)
    print('CONFIGURATION')
    print('='*60)
    print(OmegaConf.to_yaml(cfg))
    print('='*60 + '\n')
    
    # Extract model-specific parameters
    if cfg.model.type == 'deepsets':
        num_phi_layers = cfg.model.num_phi_layers
        num_rho_layers = cfg.model.num_rho_layers
        num_blocks = None
        num_heads = None
        pool_mode = cfg.model.get('pool_mode', 'sum')
    else:  # transformer
        num_phi_layers = None
        num_rho_layers = None
        num_blocks = cfg.model.num_blocks
        num_heads = cfg.model.num_heads
        pool_mode = 'sum'
    
    # Extract symmetry parameters
    symmetry_enabled = cfg.get('symmetry', {}).get('enabled', False)
    symmetry_layer = cfg.get('symmetry', {}).get('layer_idx', -1) if symmetry_enabled else None
    
    return run_training(
        # Data params
        num_events=cfg.data.num_events,
        n_particles=cfg.data.n_particles,
        batch_size=cfg.data.batch_size,
        input_scale=cfg.data.input_scale,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        # Training params
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_epochs=cfg.training.warmup_epochs,
        weight_decay=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        dropout=cfg.training.dropout,
        early_stopping_patience=cfg.training.get('early_stopping_patience', None),
        # Model params
        model_type=cfg.model.type,
        hidden_channels=cfg.model.hidden_channels,
        num_phi_layers=num_phi_layers,
        num_rho_layers=num_rho_layers,
        pool_mode=pool_mode,
        num_blocks=num_blocks,
        num_heads=num_heads,
        use_mean_pooling=cfg.model.get('use_mean_pooling', True),
        dropout_prob=cfg.model.get('dropout_prob'),
        multi_query=cfg.model.get('multi_query', False),
        increase_hidden_channels=cfg.model.get('increase_hidden_channels', 1),
        checkpoint_blocks=cfg.model.get('checkpoint_blocks', False),
        # Symmetry params
        symmetry_enabled=symmetry_enabled,
        symmetry_layer=symmetry_layer,
        lambda_sym_max=cfg.get('symmetry', {}).get('lambda_sym_max', 1.0),
        std_eta=cfg.get('symmetry', {}).get('std_eta', 0.5),
        # Other
        run_seed=cfg.run_seed,
        headless=cfg.headless,
    )


if __name__ == '__main__':
    main()

