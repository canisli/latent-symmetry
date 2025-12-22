import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from models import MLP, so3_orbit_variance_loss
from data import ScalarFieldDataset
from plots import plot_losses, visualize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, loss_fn, loader, optimizer, symmetry_layer=None, lambda_sym=0.0):
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
            sym_loss = so3_orbit_variance_loss(model, xb, symmetry_layer)
        
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

def evaluate(model, loss_fn, loader, symmetry_layer=None):
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
                sym_loss += so3_orbit_variance_loss(model, xb, symmetry_layer).item()
        task_loss /= len(loader)
        if symmetry_layer is not None:
            sym_loss /= len(loader)
        return task_loss, sym_loss

def main(headless=False, symmetry_layer=-1, lambda_sym_max=1.0, lambda_sym_min=0.0, learning_rate=3e-4, seed=42):
    """
    Main training function.
    
    Args:
        headless: If True, skip plotting and visualization
        symmetry_layer: Layer index for symmetry loss (None to disable)
        lambda_sym_max: Maximum lambda_sym value for 1-cosine schedule
        lambda_sym_min: Minimum lambda_sym value for 1-cosine schedule
        learning_rate: Learning rate for optimizer
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with keys: learning_rate, lambda_sym_max, lambda_sym_min, 
        symmetry_layer, test_task_loss, test_sym_loss
    """
    set_seed(seed)
    
    field = ScalarFieldDataset(10000, seed=seed)
    batch_size = 128
    
    # Create generator for reproducible random split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(field, [0.6, 0.2, 0.2], generator=generator)
    
    # Create generator for reproducible DataLoader shuffling
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLP(3, [128, 128, 128, 128, 128, 128], 1).to(device)
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
        
        train_task_loss, train_sym_loss, task_batch_losses, sym_batch_losses = train(model, loss_fn, train_loader, optimizer, symmetry_layer, lambda_sym)
        train_task_batch_losses.extend(task_batch_losses)
        train_sym_batch_losses.extend(sym_batch_losses)
        val_task_loss, val_sym_loss = evaluate(model, loss_fn, val_loader, symmetry_layer)

        
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

    test_task_loss, test_sym_loss = evaluate(model, loss_fn, test_loader, symmetry_layer)
    
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
    main()