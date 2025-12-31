import torch
import matplotlib.pyplot as plt
from data import compute_scalar_field

def plot_losses(train_task_losses, val_task_losses, train_task_batch_losses=None,
                train_sym_batch_losses=None, train_sym_losses=None, val_sym_losses=None,
                lambda_sym_values=None):
    # Create two subplots: one for task loss, one for symmetry loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # ===== Task Loss Plot =====
    # Plot batch losses behind epoch losses if provided
    if train_task_batch_losses is not None:
        n_batches = len(train_task_batch_losses)
        n_epochs = len(train_task_losses)
        batch_x = torch.linspace(0, n_epochs - 1, n_batches).numpy()
        ax1.plot(batch_x, train_task_batch_losses, alpha=0.3, color='k', linewidth=0.5, label='train (batch)')
    
    # Plot epoch losses on top
    ax1.plot(train_task_losses, label='train (epoch)', linewidth=1, color='k')
    ax1.plot(val_task_losses, label='val', linewidth=1, color='b')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Task Loss', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_yscale('log')
    ax1.set_title('Task Loss')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # ===== Symmetry Loss Plot =====
    # Plot batch symmetry losses if provided
    if train_sym_batch_losses is not None and len(train_sym_batch_losses) > 0:
        n_batches = len(train_sym_batch_losses)
        n_epochs = len(train_sym_losses) if train_sym_losses else len(val_sym_losses) if val_sym_losses else 1
        batch_x = torch.linspace(0, n_epochs - 1, n_batches).numpy()
        ax2.plot(batch_x, train_sym_batch_losses, alpha=0.3, color='g', linewidth=0.5, label='train sym (batch)')
    
    # Plot epoch symmetry losses
    if train_sym_losses is not None:
        ax2.plot(train_sym_losses, label='train sym (epoch)', linewidth=1, color='g')
    if val_sym_losses is not None:
        ax2.plot(val_sym_losses, label='val sym', linewidth=1, color='c')
    
    # Plot lambda_sym on secondary y-axis if provided
    if lambda_sym_values:
        ax2_lambda = ax2.twinx()
        epochs = range(len(lambda_sym_values))
        ax2_lambda.plot(epochs, lambda_sym_values, label='λ_sym', linewidth=1.5, color='r', linestyle=':')
        ax2_lambda.set_ylabel('λ_sym', color='k')
        ax2_lambda.tick_params(axis='y', labelcolor='k')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Symmetry Loss', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    if lambda_sym_values:
        ax2.set_yscale('log')
    ax2.set_title('Symmetry Loss')
    
    # Combine legends for symmetry plot
    lines1, labels1 = ax2.get_legend_handles_labels()
    if lambda_sym_values is not None:
        lines2, labels2 = ax2_lambda.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax2.legend(loc='best')
    ax2.grid(alpha=0.3)


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
    

    # plt.savefig('2d_slice_visualization.png', dpi=150, bbox_inches='tight')