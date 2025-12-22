import torch
import matplotlib.pyplot as plt
from data import compute_scalar_field

def plot_losses(train_losses, val_losses, train_batch_losses=None):
    plt.figure(figsize=(10, 5))
    
    # Plot batch losses behind epoch losses if provided
    if train_batch_losses is not None:
        # Calculate x positions for batch losses (normalized to epoch scale)
        n_batches = len(train_batch_losses)
        n_epochs = len(train_losses)
        batch_x = torch.linspace(0, n_epochs - 1, n_batches).numpy()
        plt.plot(batch_x, train_batch_losses, alpha=0.3, color='k', linewidth=0.5, label='train (batch)')
    
    # Plot epoch losses on top
    plt.plot(train_losses, label='train (epoch)', linewidth=1, color='k')
    plt.plot(val_losses, label='val', linewidth=1, color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')


def visualize(model, device):
    # Create 2D slice at z=0
    n_points = 100
    x = torch.linspace(-3, 3, n_points)
    y = torch.linspace(-3, 3, n_points)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    zz = torch.zeros_like(xx)
    X = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().reshape(n_points, n_points)
    
    # Get true values using the same function as ScalarFieldDataset
    y_true = compute_scalar_field(X.cpu()).reshape(n_points, n_points)
    
    # Compute difference
    y_diff = y_pred - y_true
    
    # Create 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot model prediction
    im1 = axes[0].imshow(y_pred.numpy(), extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
    axes[0].set_title('Model Prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot true value
    im2 = axes[1].imshow(y_true.numpy(), extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
    axes[1].set_title('True Value')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference
    im3 = axes[2].imshow(y_diff.numpy(), extent=[-5, 5, -5, 5], origin='lower', cmap='RdBu_r', vmin=-y_diff.abs().max(), vmax=y_diff.abs().max())
    axes[2].set_title('Difference (Pred - True)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    

    plt.savefig('2d_slice_visualization.png', dpi=150, bbox_inches='tight')