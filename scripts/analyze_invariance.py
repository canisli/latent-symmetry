#!/usr/bin/env python3

from so2toy.models import MLP
from so2toy.data import create_dataloaders, gaussian_ring, x_field
from so2toy.train import train_loop, create_scheduler, plot_loss_curves
from so2toy.eval import plot_regression_surface, compute_all_Q, plot_Q_vs_layer

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Get full dataset as tensor
    X, _ = full_dataset.get_numpy()
    X = torch.tensor(X, dtype=torch.float32)
    
    # Compute Q for all layers
    print("\nComputing SO(2) invariance metrics...")
    Q_values = compute_all_Q(model, X, n_rotations=32, explained_variance=0.95, device=device)
    
    print("\nQ values by layer:")
    for layer, Q in Q_values.items():
        print(f"  {layer}: Q = {Q:.4f}")
    
    # Plot results
    plot_loss_curves(history, exp_dir / 'loss_curves.png')
    plot_regression_surface(model, full_dataset, exp_dir / 'regression_surface.png', device)
    plot_Q_vs_layer(Q_values, exp_dir / 'Q_vs_layer.png')
    
    return Q_values


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run both experiments
    Q_inv = run_experiment(cfg, gaussian_ring, "inv", output_dir, device)
    Q_noninv = run_experiment(cfg, x_field, "noninv", output_dir, device)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (Q_values, title) in zip(axes, [(Q_inv, "Invariant (Gaussian Ring)"), (Q_noninv, "Non-invariant (x)")]):
        layers = list(Q_values.keys())
        values = list(Q_values.values())
        x = range(len(layers))
        ax.bar(x, values, color='steelblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Q (Orbit Variance)')
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Q_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
