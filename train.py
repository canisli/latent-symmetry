import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import hashlib
import argparse

from models import MLP
from symmetry import so3_orbit_variance_loss
from data import ScalarFieldDataset
from plots import plot_losses, visualize

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

def train(model, loss_fn, loader, optimizer, symmetry_layer=None, lambda_sym=0.0, augmentation_generator=None):
    task_batch_losses = []
    sym_batch_losses = []
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        task_loss = loss_fn(pred, yb)
        
        # Compute symmetry loss if enabled
        sym_loss = 0.0
        if symmetry_layer is not None and lambda_sym > 0:
            sym_loss = so3_orbit_variance_loss(model, xb, symmetry_layer, generator=augmentation_generator)
        
        # Track losses separately
        task_batch_losses.append(task_loss.item())
        sym_batch_losses.append(sym_loss.item() if isinstance(sym_loss, torch.Tensor) else sym_loss)
        
        # Total loss for backprop
        total_loss = task_loss + lambda_sym * sym_loss
        total_loss.backward()
        optimizer.step()
    
    avg_task_loss = sum(task_batch_losses) / len(task_batch_losses)
    avg_sym_loss = sum(sym_batch_losses) / len(sym_batch_losses) if sym_batch_losses else 0.0
    
    return avg_task_loss, avg_sym_loss, task_batch_losses, sym_batch_losses

def evaluate(model, loss_fn, loader, symmetry_layer=None, augmentation_generator=None):
    model.eval()
    with torch.no_grad():
        task_loss = 0
        sym_loss = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            task_loss += loss_fn(pred, yb).item()
            
            if symmetry_layer is not None:
                sym_loss += so3_orbit_variance_loss(model, xb, symmetry_layer, generator=augmentation_generator).item()
        task_loss /= len(loader)
        if symmetry_layer is not None:
            sym_loss /= len(loader)
        return task_loss, sym_loss

def main(headless=False, symmetry_layer=-1, lambda_sym_max=1.0, learning_rate=3e-4, num_hidden_layers=6, hidden_dim=128, run_seed=42):
    """
    Main training function.
    
    Args:
        headless: If True, skip plotting and visualization
        symmetry_layer: Layer index for symmetry loss (None to disable)
        lambda_sym_max: Maximum lambda_sym value for 1-cosine schedule (min is always 0.0)
        learning_rate: Learning rate for optimizer
        num_hidden_layers: Number of hidden layers
        hidden_dim: Size of each hidden layer (default: 128)
        run_seed: Base random seed for reproducibility (derives data, model, and augmentation seeds)
    
    Returns:
        Dictionary with keys: learning_rate, lambda_sym_max, lambda_sym_min, 
        symmetry_layer, test_task_loss, test_sym_loss
    """
    lambda_sym_min = 0.0  # Always zero
    
    # Derive seeds for different randomness sources
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    augmentation_seed = derive_seed(run_seed, "augmentation")
    
    field = ScalarFieldDataset(10000, seed=data_seed)
    batch_size = 128
    
    # Create generator for reproducible random split
    generator = torch.Generator().manual_seed(data_seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(field, [0.6, 0.2, 0.2], generator=generator)
    
    # Create generator for reproducible DataLoader shuffling
    loader_generator = torch.Generator().manual_seed(data_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Set model seed before model creation
    set_model_seed(model_seed)
    
    # Create hidden_dims list: 
    # Convention: [hidden_dim, hidden_dim, ...] = num_hidden_layers hidden layers
    # For num_hidden_layers=4, hidden_dim=128: [128, 128, 128, 128] -> 4 hidden layers
    # For num_hidden_layers=6, hidden_dim=128: [128, 128, 128, 128, 128, 128] -> 6 hidden layers
    hidden_dims = [hidden_dim] * num_hidden_layers
    dims = [3, *hidden_dims, 1]
    model = MLP(dims).to(device)
    
    # Create augmentation generator (will be moved to device when used)
    augmentation_generator = torch.Generator(device=device).manual_seed(augmentation_seed)
    loss_fn = torch.nn.MSELoss()
    lr = learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    num_epochs = 100
    
    pbar = tqdm(range(num_epochs))
    train_task_losses = []
    val_task_losses = []
    train_task_batch_losses = []
    train_sym_batch_losses = []
    lambda_sym_values = []
    train_sym_losses = []
    val_sym_losses = []
    
    for epoch in pbar:
        # Compute lambda_sym with 1-cosine schedule (min to max)
        # lambda(epoch) = lambda_min + (lambda_max - lambda_min) * (1 - cos(π * epoch / max_epochs)) / 2
        progress = epoch / max(num_epochs - 1, 1)  # Normalize to [0, 1]
        lambda_sym = lambda_sym_min + (lambda_sym_max - lambda_sym_min) * (1 - np.cos(np.pi * progress)) / 2
        lambda_sym_values.append(lambda_sym)
        
        train_task_loss, train_sym_loss, task_batch_losses, sym_batch_losses = train(model, loss_fn, train_loader, optimizer, symmetry_layer, lambda_sym, augmentation_generator)
        train_task_batch_losses.extend(task_batch_losses)
        train_sym_batch_losses.extend(sym_batch_losses)
        val_task_loss, val_sym_loss = evaluate(model, loss_fn, val_loader, symmetry_layer, augmentation_generator)

        
        train_task_losses.append(train_task_loss)
        val_task_losses.append(val_task_loss)
        avg_train_sym_loss = sum(sym_batch_losses) / len(sym_batch_losses) if sym_batch_losses else 0.0
        train_sym_losses.append(avg_train_sym_loss)
        val_sym_losses.append(val_sym_loss)
        
        pbar.set_postfix({
            'task': f'{train_task_loss:.2e}', 
            'sym': f'{train_sym_loss:.2e}',
            'val': f'{val_task_loss:.2e}',
        })

    test_task_loss, test_sym_loss = evaluate(model, loss_fn, test_loader, symmetry_layer, augmentation_generator)
    
    # Print data section
    print('\n' + '='*25)
    print('DATA')
    print('='*25)
    print(f'Number of data points: {len(field)}')
    print(f'Number of epochs:      {len(train_task_losses)}')
    print(f'Functional form:       {field.functional_form}')
    print('='*25)
    
    # Print configuration section
    print('\n' + '='*25)
    print('CONFIGURATION')
    print('='*25)
    print(f'Learning rate:     {lr:.2e}')
    print(f'Symmetry layer:     {symmetry_layer}')
    print(f'Lambda symmetry:   {lambda_sym_min:.4f} -> {lambda_sym_max:.4f}')
    print('='*25)
    
    # Print results section
    print('\n' + '='*25)
    print('RESULTS')
    print('='*25)
    print(f'Test task loss:     {test_task_loss:.4e}')
    print(f'Test symmetry loss: {f"{test_sym_loss:.4e}" if symmetry_layer is not None else "N/A"}')

    # Only plot if not in headless mode
    if not headless:
        plot_losses(train_task_losses, val_task_losses, train_task_batch_losses, 
                    train_sym_batch_losses, train_sym_losses, val_sym_losses, lambda_sym_values)
        visualize(model, device)
        plt.show()
    
    # Return results dictionary
    return {
        'learning_rate': lr,
        'lambda_sym_max': lambda_sym_max,
        'lambda_sym_min': lambda_sym_min,
        'symmetry_layer': symmetry_layer,
        'test_task_loss': test_task_loss,
        'test_sym_loss': test_sym_loss if symmetry_layer is not None else None
    }


if __name__ == '__main__':
    def symmetry_layer_type(value):
        """Convert string to symmetry_layer value (int, None, or -1)."""
        if value.lower() == 'none':
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"symmetry_layer must be an integer or 'None', got '{value}'")
    
    parser = argparse.ArgumentParser(description='Train MLP model with symmetry regularization')
    parser.add_argument('--symmetry-layer', type=symmetry_layer_type, default=-1,
                        help='Layer index for symmetry loss (-1 for last layer, "None" to disable)')
    parser.add_argument('--lambda-sym-max', type=float, default=1.0,
                        help='Maximum lambda_sym value for 1-cosine schedule (default: 1.0)')
    parser.add_argument('--num-hidden-layers', type=int, default=6,
                        help='Number of hidden layers (default: 6)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Size of each hidden layer (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for optimizer (default: 3e-4)')
    parser.add_argument('--run-seed', type=int, default=42,
                        help='Base random seed for reproducibility (default: 42)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (skip plotting and visualization)')
    
    args = parser.parse_args()
    
    main(
        headless=args.headless,
        symmetry_layer=args.symmetry_layer,
        lambda_sym_max=args.lambda_sym_max,
        learning_rate=args.learning_rate,
        num_hidden_layers=args.num_hidden_layers,
        hidden_dim=args.hidden_dim,
        run_seed=args.run_seed
    )