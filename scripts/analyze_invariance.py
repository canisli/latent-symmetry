#!/usr/bin/env python3
"""
Analyze invariance of trained models using configurable metrics.

Trains models on invariant and non-invariant tasks, then computes
and plots selected symmetry metrics for each layer.
"""

from latsym.models import MLP
from latsym.tasks import create_dataloaders, gaussian_ring, x_field
from latsym.train import train_loop, create_scheduler, plot_loss_curves
from latsym.eval import plot_regression_surface
from latsym.metrics import get_metric, list_metrics

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json


def build_model(cfg: DictConfig) -> nn.Module:
    dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.num_layers + [cfg.output_dim]
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    return MLP(dims=dims, act=act_map.get(cfg.activation, nn.ReLU))


def run_experiment(cfg: DictConfig, scalar_field_fn, name: str, output_dir: Path, device: torch.device):
    """Run a single experiment with the given scalar field."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")
    
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    seed = cfg.experiment.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader, val_loader, full_dataset = create_dataloaders(
        n_samples=cfg.data.n_samples,
        r_min=cfg.data.r_min,
        r_max=cfg.data.r_max,
        scalar_field_fn=scalar_field_fn,
        train_split=cfg.data.train_split,
        batch_size=cfg.train.batch_size,
        seed=seed,
    )
    
    model = build_model(cfg.model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = create_scheduler(optimizer, cfg.train.total_steps, cfg.train.warmup_steps) if cfg.train.use_scheduler else None
    
    history = train_loop(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, device,
        cfg.train.total_steps, cfg.train.log_interval, cfg.train.eval_interval,
        exp_dir, cfg.train.save_best,
    )
    
    print(f"Final Train MSE: {history['train_loss'][-1]:.6f}")
    print(f"Final Val MSE: {history['val_loss'][-1]:.6f}")
    print(f"Final Val MAE: {history['val_mae'][-1]:.4f}")
    
    best_model_path = exp_dir / 'model_best.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    # Get full dataset as tensor
    X, _ = full_dataset.get_numpy()
    X = torch.tensor(X, dtype=torch.float32)
    
    # Compute all enabled metrics
    all_metric_values = {}
    
    for metric_name in cfg.metrics.enabled:
        print(f"\nComputing {metric_name} metric...")
        
        # Get metric-specific config
        metric_cfg = OmegaConf.to_container(cfg.metrics.get(metric_name, {}), resolve=True)
        metric = get_metric(metric_name, **metric_cfg)
        
        # Compute metric
        values = metric.compute(model, X, device=device)
        all_metric_values[metric_name] = values
        
        # Print values
        print(f"\n{metric_name} values by layer:")
        for layer, val in values.items():
            print(f"  {layer}: {metric_name} = {val:.4f}")
        
        # Plot
        metric.plot(values, exp_dir / f'{metric_name}_vs_layer.png')
    
    # Save all metric values to JSON
    with open(exp_dir / 'metric_values.json', 'w') as f:
        json.dump(all_metric_values, f, indent=2)
    
    # Plot standard visualizations
    plot_loss_curves(history, exp_dir / 'loss_curves.png')
    plot_regression_surface(model, full_dataset, exp_dir / 'regression_surface.png', device)
    plt.close('all')
    
    return all_metric_values


def plot_metric_comparison(metric_values_inv, metric_values_noninv, metric_name, save_path):
    """Plot comparison of a metric between invariant and non-invariant models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (values, title) in zip(axes, [
        (metric_values_inv, "Invariant (Gaussian Ring)"),
        (metric_values_noninv, "Non-invariant (x)")
    ]):
        layers = list(values.keys())
        vals = list(values.values())
        x = range(len(layers))
        ax.bar(x, vals, color='steelblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_xlabel('Layer')
        ax.set_ylabel(f'{metric_name}')
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        if metric_name == "Q":
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label=f'{metric_name}=1')
            ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"\nAvailable metrics: {list_metrics()}")
    print(f"Enabled metrics: {list(cfg.metrics.enabled)}")
    
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run both experiments
    metrics_inv = run_experiment(cfg, gaussian_ring, "inv", output_dir, device)
    metrics_noninv = run_experiment(cfg, x_field, "noninv", output_dir, device)
    
    # Plot comparison for each metric
    for metric_name in cfg.metrics.enabled:
        if metric_name in metrics_inv and metric_name in metrics_noninv:
            plot_metric_comparison(
                metrics_inv[metric_name],
                metrics_noninv[metric_name],
                metric_name,
                output_dir / f'{metric_name}_comparison.png'
            )
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
