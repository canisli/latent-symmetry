#!/usr/bin/env python3
"""
Analyze invariance of trained models using configurable metrics.

Trains models on invariant and non-invariant tasks, then computes
and plots selected symmetry metrics for each layer.
"""

from latsym.models import MLP
from latsym.tasks import create_dataloaders, gaussian_ring, x_field, fourier, mix
from latsym.train import train_loop, create_scheduler, plot_loss_curves
from latsym.eval import plot_regression_surface
from latsym.metrics import get_metric, list_metrics
from latsym.metrics.q_metric import compute_oracle_Q

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from typing import Callable, Optional


def get_field_fn(field_name: str, field_args: dict = None) -> Callable:
    """
    Get a scalar field function by name.
    
    Args:
        field_name: One of 'gaussian_ring', 'x_field', 'fourier', 'mix'
        field_args: Optional dict of arguments for parameterized fields:
                   - fourier: {'k': int} (frequency)
                   - mix: {'alpha': float} (mixing parameter 0-1)
    
    Returns:
        Scalar field function with signature (x, y, r) -> values
    
    Raises:
        ValueError: If field_name is unknown
    """
    field_args = field_args or {}
    
    if field_name == "gaussian_ring":
        return gaussian_ring
    elif field_name == "x_field":
        return x_field
    elif field_name == "fourier":
        k = field_args.get("k", 2)
        return fourier(k=k)
    elif field_name == "mix":
        alpha = field_args.get("alpha", 0.5)
        return mix(alpha=alpha)
    else:
        raise ValueError(f"Unknown field: {field_name}. "
                        f"Available: gaussian_ring, x_field, fourier, mix")


def build_model(cfg: DictConfig) -> nn.Module:
    dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.num_layers + [cfg.output_dim]
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    return MLP(dims=dims, act=act_map.get(cfg.activation, nn.ReLU))


def run_single_field(cfg: DictConfig, scalar_field_fn, device: torch.device,
                     output_dir: Path = None, name: str = None):
    """
    Train on a single field and return Q values + oracle Q.
    
    Args:
        cfg: Hydra config with data, model, train, metrics sections
        scalar_field_fn: Function (x, y, r) -> scalar field values
        device: torch device
        output_dir: If provided, save all artifacts here. If None, use temp directory.
        name: Run name for logging (optional)
    
    Returns:
        Tuple of (Q_values dict, oracle_Q float, history dict)
    """
    import tempfile
    from latsym.metrics.q_metric import plot_Q_vs_layer
    
    if name:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
    
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
    
    # Use output_dir if provided, otherwise temp directory
    use_temp = output_dir is None
    if use_temp:
        temp_ctx = tempfile.TemporaryDirectory()
        save_dir = Path(temp_ctx.__enter__())
    else:
        temp_ctx = None
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        history = train_loop(
            model, train_loader, val_loader, loss_fn, optimizer, scheduler, device,
            cfg.train.total_steps, cfg.train.log_interval, cfg.train.eval_interval,
            save_dir, cfg.train.save_best,
        )
        
        print(f"Final Train MSE: {history['train_loss'][-1]:.6f}")
        print(f"Final Val MSE: {history['val_loss'][-1]:.6f}")
        print(f"Final Val MAE: {history['val_mae'][-1]:.4f}")
        
        best_model_path = save_dir / 'model_best.pt'
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        
        # Get full dataset as tensor
        X, y = full_dataset.get_numpy()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)
        
        # Compute Q metric
        metric_cfg = OmegaConf.to_container(cfg.metrics.get("Q", {}), resolve=True)
        metric = get_metric("Q", **metric_cfg)
        Q_values = metric.compute(model, X, device=device)
        
        # Compute oracle Q
        oracle_Q = compute_oracle_Q(X, y, scalar_field_fn, n_rotations=32, device=device)
        
        print(f"  Oracle Q = {oracle_Q:.4f}")
        print(f"  Q values by layer:")
        for layer, val in Q_values.items():
            print(f"    {layer}: Q = {val:.4f}")
        
        # Save artifacts if output_dir provided
        if not use_temp:
            # Save metric values
            all_metric_values = {"Q": Q_values}
            with open(save_dir / 'metric_values.json', 'w') as f:
                json.dump(all_metric_values, f, indent=2)
            
            # Save Q plot
            plot_Q_vs_layer(Q_values, save_dir / 'Q_vs_layer.png', oracle_Q=oracle_Q)
            
            # Save loss curves
            plot_loss_curves(history, save_dir / 'loss_curves.png')
            
            # Save regression surface
            plot_regression_surface(model, full_dataset, save_dir / 'regression_surface.png', device)
            
            plt.close('all')
        
        return Q_values, oracle_Q, history
        
    finally:
        if temp_ctx is not None:
            temp_ctx.__exit__(None, None, None)


def run_batch_mode(cfg: DictConfig, output_dir: Path, device: torch.device):
    """
    Run experiments specified in a YAML file.
    
    Each run can specify:
        - name: Run name (used for folder and aggregated files)
        - field: Field type (gaussian_ring, x_field, fourier, mix)
        - field_args: Optional args for parameterized fields (k, alpha)
        - model: Optional model config overrides (num_layers, hidden_dim, etc.)
        - train: Optional training config overrides (total_steps, learning_rate, etc.)
        - data: Optional data config overrides (n_samples, etc.)
    """
    import yaml
    
    # Load runs specification
    runs_file = cfg.experiment.get("runs_file", "config/runs/default.yaml")
    runs_path = Path(runs_file)
    
    if not runs_path.exists():
        # Try relative to workspace root
        runs_path = Path(hydra.utils.get_original_cwd()) / runs_file
    
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs file not found: {runs_file}")
    
    print(f"Loading runs from: {runs_path}")
    
    with open(runs_path, 'r') as f:
        runs_config = yaml.safe_load(f)
    
    runs = runs_config.get('runs', [])
    print(f"Found {len(runs)} run(s) to execute\n")
    
    # Track Q plots for aggregation
    q_plot_sources = []
    
    for run_spec in runs:
        run_name = run_spec.get('name', 'unnamed')
        field_name = run_spec.get('field', 'gaussian_ring')
        field_args = run_spec.get('field_args', {})
        
        # Get the field function
        field_fn = get_field_fn(field_name, field_args)
        
        # Create merged config with run-specific overrides
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        
        # Apply model overrides
        if 'model' in run_spec:
            for key, value in run_spec['model'].items():
                run_cfg.model[key] = value
        
        # Apply train overrides
        if 'train' in run_spec:
            for key, value in run_spec['train'].items():
                run_cfg.train[key] = value
        
        # Apply data overrides
        if 'data' in run_spec:
            for key, value in run_spec['data'].items():
                run_cfg.data[key] = value
        
        # Create run output directory
        run_dir = output_dir / run_name
        
        # Run the experiment
        Q_values, oracle_Q, history = run_single_field(
            run_cfg, field_fn, device, 
            output_dir=run_dir, 
            name=run_name
        )
        
        # Track for aggregation
        q_plot_src = run_dir / 'Q_vs_layer.png'
        if q_plot_src.exists():
            q_plot_sources.append((run_name, q_plot_src))
    
    # Create hard links for Q plots in root directory
    print(f"\n{'='*60}")
    print("Aggregating Q plots to root directory")
    print("="*60)
    
    for run_name, src_path in q_plot_sources:
        dest_path = output_dir / f"{run_name}_Q_vs_layer.png"
        try:
            # Remove existing link/file if present
            if dest_path.exists():
                dest_path.unlink()
            # Create hard link
            os.link(src_path, dest_path)
            print(f"  Linked: {dest_path.name}")
        except OSError as e:
            # Fall back to copy if hard link fails (e.g., cross-device)
            import shutil
            shutil.copy2(src_path, dest_path)
            print(f"  Copied: {dest_path.name} (hard link failed: {e})")
    
    print(f"\nAll results saved to: {output_dir}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"\nAvailable metrics: {list_metrics()}")
    print(f"Enabled metrics: {list(cfg.metrics.enabled)}")
    
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_batch_mode(cfg, output_dir, device)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
