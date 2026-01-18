import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import hashlib
import hydra
import yaml
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple

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

def load_efp_preset(preset_name: str, config_dir: str = "config") -> List[List[Tuple[int, int]]]:
    """
    Load EFP preset from config file.
    
    Args:
        preset_name: Name of the preset (e.g., 'deg3')
        config_dir: Directory containing efp_presets.yaml
    
    Returns:
        List of edge configurations, where each edge config is a list of tuples
    """
    preset_file = Path(config_dir) / "efp_presets.yaml"
    
    if not preset_file.exists():
        raise FileNotFoundError(f"EFP presets file not found: {preset_file}")
    
    with open(preset_file, 'r') as f:
        presets = yaml.safe_load(f)
    
    if preset_name not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"EFP preset '{preset_name}' not found. Available presets: {available}")
    
    preset = presets[preset_name]
    edges_raw = preset['edges']
    
    # Convert from list format [[0,1], [0,1], [0,1]] to tuple format [(0,1), (0,1), (0,1)]
    edges_list = []
    for edge_config in edges_raw:
        edges_list.append([tuple(edge) for edge in edge_config])
    
    return edges_list

def build_dataloader(
    n_events: int,
    n_particles: int,
    batch_size: int,
    input_scale: float,
    edges_list: List[List[Tuple[int, int]]],
    measure: str = 'kinematic',
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
    target_transform: str = 'log1p',
):
    """Build dataloader with specified edges_list and measure configuration."""
    return make_kp_dataloader(
        edges_list=edges_list,
        n_events=n_events,
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        measure=measure,
        beta=beta,
        kappa=kappa,
        normed=normed,
        target_transform=target_transform,
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
        Tuple of (task_loss, rel_rmse, sym_loss, per_kp_rel_rmse, per_kp_rmse, per_kp_std)
        where:
        - per_kp_rel_rmse: list of relative RMSE values for each KP
        - per_kp_rmse: list of absolute RMSE values for each KP
        - per_kp_std: list of standard deviations of targets for each KP
    """
    model.eval()
    with torch.no_grad():
        task_loss = 0
        sym_loss = 0
        
        # Per-KP statistics
        num_kps = None
        per_kp_sum_sq_error = None
        per_kp_sum_y = None
        per_kp_sum_y_sq = None
        per_kp_count = None
        
        for xb, yb in loader:
            xb = xb.to(device)  # (batch_size, num_particles, 4)
            yb = yb.to(device)  # (batch_size, 1, num_kps)
            
            # Squeeze the singleton dimension from yb
            yb = yb.squeeze(1)

            pred = model(xb)  # (batch_size, num_kps)
            
            # Initialize per-KP accumulators on first batch
            if num_kps is None:
                num_kps = yb.shape[1]
                per_kp_sum_sq_error = [0.0] * num_kps
                per_kp_sum_y = [0.0] * num_kps
                per_kp_sum_y_sq = [0.0] * num_kps
                per_kp_count = [0] * num_kps
            
            task_loss += loss_fn(pred, yb).item()
            
            if symmetry_layer is not None:
                sym_loss += lorentz_orbit_variance_loss(
                    model, xb, symmetry_layer,
                    std_eta=std_eta,
                    generator=augmentation_generator,
                ).item()
            
            diff = pred - yb
            
            # Accumulate per-KP statistics
            for kp_idx in range(num_kps):
                kp_diff = diff[:, kp_idx]
                kp_y = yb[:, kp_idx]
                per_kp_sum_sq_error[kp_idx] += kp_diff.pow(2).sum().item()
                per_kp_sum_y[kp_idx] += kp_y.sum().item()
                per_kp_sum_y_sq[kp_idx] += kp_y.pow(2).sum().item()
                per_kp_count[kp_idx] += kp_y.numel()
        
        task_loss /= len(loader)
        if symmetry_layer is not None:
            sym_loss /= len(loader)
        
        # Compute per-KP metrics
        per_kp_rel_rmse = []
        per_kp_rmse = []
        per_kp_std = []
        for kp_idx in range(num_kps):
            # Relative RMSE
            _, kp_rel_rmse = _accumulate_regression_stats(
                per_kp_sum_sq_error[kp_idx],
                per_kp_sum_y[kp_idx],
                per_kp_sum_y_sq[kp_idx],
                per_kp_count[kp_idx]
            )
            per_kp_rel_rmse.append(kp_rel_rmse)
            
            # Absolute RMSE
            kp_mse = per_kp_sum_sq_error[kp_idx] / per_kp_count[kp_idx] if per_kp_count[kp_idx] > 0 else 0.0
            kp_rmse = math.sqrt(kp_mse)
            per_kp_rmse.append(kp_rmse)
            
            # Standard deviation of targets
            kp_mean = per_kp_sum_y[kp_idx] / per_kp_count[kp_idx] if per_kp_count[kp_idx] > 0 else 0.0
            kp_variance = (per_kp_sum_y_sq[kp_idx] / per_kp_count[kp_idx] - kp_mean ** 2) if per_kp_count[kp_idx] > 0 else 0.0
            kp_std = math.sqrt(max(kp_variance, 0.0))
            per_kp_std.append(kp_std)
        
        # Overall relative RMSE is the mean of per-KP relative RMSE values
        rel_rmse = np.mean(per_kp_rel_rmse) if per_kp_rel_rmse else float("nan")
        
        return task_loss, rel_rmse, sym_loss, per_kp_rel_rmse, per_kp_rmse, per_kp_std

def run_training(
    # Data params
    num_events: int = 10000,
    n_particles: int = 128,
    batch_size: int = 256,
    input_scale: float = 0.9515689,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    edges_list: List[List[Tuple[int, int]]] = None,
    # Target type params
    target_type: str = 'kinematic',  # 'kinematic' (Lorentz invariant) or 'efp' (non-invariant)
    efp_measure: str = 'eeefm',
    efp_beta: float = 2.0,
    efp_kappa: float = 1.0,
    efp_normed: bool = False,  # If True, normalize energies (usually False for non-invariant)
    target_transform: str = 'log1p',  # 'log1p', 'log_standardized', or 'standardized'
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
    save_model_path: str = None,
):
    """
    Core training function that can be called directly.
    
    Args:
        target_type: 'kinematic' for Lorentz-invariant KPs, 'efp' for non-invariant EFPs
        efp_measure: EFP measure to use when target_type='efp' (default: 'eeefm')
        efp_beta: Angular weighting exponent for EFPs
        efp_kappa: Energy weighting exponent for EFPs
    
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
    
    # Determine measure parameters based on target type
    if target_type == 'kinematic':
        measure = 'kinematic'
        beta = 2.0  # ignored by kinematic measure
        kappa = 1.0  # ignored by kinematic measure
        normed = False  # ignored by kinematic measure
    elif target_type == 'efp':
        measure = efp_measure
        beta = efp_beta
        kappa = efp_kappa
        normed = efp_normed
    else:
        raise ValueError(f"Unknown target_type: {target_type}. Must be 'kinematic' or 'efp'")
    
    # Build dataloaders 
    train_loader = build_dataloader(
        n_events=int(num_events * train_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        edges_list=edges_list,
        measure=measure,
        beta=beta,
        kappa=kappa,
        normed=normed,
        target_transform=target_transform,
    )
    val_loader = build_dataloader(
        n_events=int(num_events * val_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        edges_list=edges_list,
        measure=measure,
        beta=beta,
        kappa=kappa,
        normed=normed,
        target_transform=target_transform,
    )
    test_loader = build_dataloader(
        n_events=int(num_events * test_split),
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        edges_list=edges_list,
        measure=measure,
        beta=beta,
        kappa=kappa,
        normed=normed,
        target_transform=target_transform,
    )
    
    # Create model
    # Input: 4 channels (E, px, py, pz)
    # Output: num_kps KP values (from edges_list)
    if edges_list is None:
        raise ValueError("edges_list must be provided to determine num_kps")
    num_kps = len(edges_list)
    
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
        val_task_loss, val_rel_rmse, val_sym_loss, val_per_kp_rel_rmse, val_per_kp_rmse, val_per_kp_std = evaluate(
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
    
    # Save model if path is provided
    if save_model_path is not None:
        save_path = Path(save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        if not headless:
            print(f'\nModel saved to: {save_path}')
    
    test_task_loss, test_rel_rmse, test_sym_loss, test_per_kp_rel_rmse, test_per_kp_rmse, test_per_kp_std = evaluate(
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
        # Show target type info
        if target_type == 'kinematic':
            print(f'Target type: kinematic (Lorentz invariant)')
        else:
            print(f'Target type: EFP ({measure}, beta={beta}, kappa={kappa}) - NOT Lorentz invariant')
        print(f'Polynomials: {len(edges_list)} graphs')
        for i, edges in enumerate(edges_list, 1):
            print(f'  {i}: {edges}')
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
        print(f'Test task loss (MSE on transformed targets): {test_task_loss:.4e}')
        print('\nPer-KP metrics:')
        print(f'{"KP":<4} {"Rel RMSE":<12} {"RMSE":<12} {"Std":<12}')
        print('-' * 40)
        for i, (rel_rmse, rmse, std) in enumerate(zip(test_per_kp_rel_rmse, test_per_kp_rmse, test_per_kp_std), 1):
            print(f'{i:<4} {rel_rmse:<12.6f} {rmse:<12.6e} {std:<12.6e}')
        print(f'\nTest relative RMSE (mean): {test_rel_rmse:.4f}')
        if symmetry_enabled:
            print(f'\nTest symmetry loss: {test_sym_loss:.4e}')
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
        
        # Collect predictions and true values for histogram plotting
        model.eval()
        all_predictions = []
        all_true_values = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yb = yb.squeeze(1)  # (batch_size, num_kps)
                pred = model(xb)  # (batch_size, num_kps)
                all_predictions.append(pred.cpu().numpy())
                all_true_values.append(yb.cpu().numpy())
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)  # (n_test_events, num_kps)
        all_true_values = np.concatenate(all_true_values, axis=0)  # (n_test_events, num_kps)
        
        # Plot histograms, one for each KP, in rows of 5
        n_cols = 5
        n_rows = math.ceil(num_kps / n_cols)
        fig_width = 4 * n_cols
        fig_height = 4 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Flatten axes to 1D for easier indexing (works for both 1D and 2D arrays)
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
        
        for kp_idx in range(num_kps):
            ax = axes[kp_idx]
            true_vals = all_true_values[:, kp_idx]
            pred_vals = all_predictions[:, kp_idx]
            
            # Determine bins based on the range of both true and predicted values
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            bins = np.linspace(min_val, max_val, 50)
            
            # Plot histograms
            ax.hist(true_vals, bins=bins, alpha=0.6, label='True', color='blue', edgecolor='black', linewidth=0.5)
            ax.hist(pred_vals, bins=bins, alpha=0.6, label='Predicted', color='red', edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('log1p(KP value)')
            ax.set_ylabel('Count')
            ax.set_title(f'KP {kp_idx + 1}\n{edges_list[kp_idx]}')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide unused subplot axes
        for kp_idx in range(num_kps, len(axes)):
            axes[kp_idx].axis('off')
        
        plt.tight_layout()
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
        'target_type': target_type,
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
    result['test_per_kp_rel_rmse'] = test_per_kp_rel_rmse
    result['test_per_kp_rmse'] = test_per_kp_rmse
    result['test_per_kp_std'] = test_per_kp_std
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
    
    # Load EFP preset from config
    efp_preset_name = cfg.data.get('efp_preset', 'deg3')
    # Use config directory relative to the script location
    script_dir = Path(__file__).parent
    config_dir = script_dir / "config"
    edges_list = load_efp_preset(efp_preset_name, str(config_dir))
    
    return run_training(
        # Data params
        num_events=cfg.data.num_events,
        n_particles=cfg.data.n_particles,
        batch_size=cfg.data.batch_size,
        input_scale=cfg.data.input_scale,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        edges_list=edges_list,
        # Target type params
        target_type=cfg.data.get('target_type', 'kinematic'),
        efp_measure=cfg.data.get('efp_measure', 'eeefm'),
        efp_beta=cfg.data.get('efp_beta', 2.0),
        efp_kappa=cfg.data.get('efp_kappa', 1.0),
        efp_normed=cfg.data.get('efp_normed', False),
        target_transform=cfg.data.get('target_transform', 'log1p'),
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
        save_model_path=cfg.get('save_model_path', None),
    )


if __name__ == '__main__':
    main()

