import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_regression_surface(model, dataset, save_path, device):
    """
    Plot predicted vs true regression surface.
    
    Creates a 3-panel figure showing:
    - Predicted field
    - True field  
    - Error (predicted - true)
    
    Args:
        model: Trained neural network model.
        dataset: Dataset with r_min, r_max, and scalar_field_fn attributes.
        save_path: Path to save the figure.
        device: Torch device.
    """
    model.eval()
    model.to(device)
    
    r_min, r_max = dataset.r_min, dataset.r_max
    margin = r_max * 0.1
    extent = r_max + margin
    
    xx, yy = np.meshgrid(np.linspace(-extent, extent, 200), np.linspace(-extent, extent, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(grid).cpu().numpy().reshape(xx.shape)
    
    true_radii = np.sqrt(xx**2 + yy**2)
    true_field = dataset.scalar_field_fn(xx, yy, true_radii)
    
    # Mask out-of-distribution points (outside the disk)
    mask = (true_radii < r_min) | (true_radii > r_max)
    preds = np.ma.masked_where(mask, preds)
    true_field = np.ma.masked_where(mask, true_field)
    error = np.ma.masked_where(mask, preds - true_field)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin, vmax = true_field.min(), true_field.max()
    for ax, data, title in zip(axes, [preds, true_field, error], 
                                ['Predicted', 'True', 'Error']):
        if title == 'Error':
            max_err = max(abs(data.min()), abs(data.max()), 0.1)
            cf = ax.contourf(xx, yy, data, levels=50, cmap='RdBu_r', vmin=-max_err, vmax=max_err)
            ax.contour(xx, yy, data, levels=[0], colors='black', linewidths=1)
        else:
            cf = ax.contourf(xx, yy, data, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.contour(xx, yy, data, levels=np.linspace(vmin, vmax, 11), colors='white', linewidths=0.5, alpha=0.7)
        plt.colorbar(cf, ax=ax)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
