import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import copy
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
    batch_losses = []
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        prediction_loss = loss_fn(pred, yb)
        
        # Add symmetry loss if enabled
        total_loss = prediction_loss
        if symmetry_layer is not None and lambda_sym > 0:
            sym_loss = so3_orbit_variance_loss(model, xb, symmetry_layer)
            total_loss = prediction_loss + lambda_sym * sym_loss
        
        batch_losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()
    train_loss = sum(batch_losses) / len(batch_losses)
    return train_loss, batch_losses

def evaluate(model, loss_fn, loader):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()
        val_loss /= len(loader)
        return val_loss

def main():
    seed = 42
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

    model = MLP(3, [128, 128, 128, 128], 1).to(device)
    loss_fn = torch.nn.MSELoss()
    lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    num_epochs = 100
    
    # Symmetry loss configuration
    symmetry_layer = None
    lambda_sym = 0.0
    
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    train_batch_losses = []
    
    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 100
    best_model_state = None
    
    for epoch in pbar:
        train_loss, batch_losses = train(model, loss_fn, train_loader, optimizer, symmetry_layer, lambda_sym)
        train_batch_losses.extend(batch_losses)
        val_loss = evaluate(model, loss_fn, val_loader)

        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model state
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
        
        pbar.set_postfix({
            'train': f'{train_loss:.2e}', 
            'val': f'{val_loss:.2e}',
        })
        
        if epochs_without_improvement >= patience:
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Restored best model with validation loss: {best_val_loss:.4e}')

    test_loss = evaluate(model, loss_fn, test_loader)
    print(f'Test loss: {test_loss:.4e}')

    plot_losses(train_losses, val_losses, train_batch_losses)
    visualize(model, device)
    plt.show()


if __name__ == '__main__':
    main()