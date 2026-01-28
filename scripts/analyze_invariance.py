#!/usr/bin/env python3
"""
Analyze invariance of trained models using configurable metrics.

Trains models on invariant and non-invariant tasks, then computes
and plots selected symmetry metrics for each layer.
"""

from latsym.models import MLP, build_model
from latsym.seeds import derive_seed, set_model_seed, set_global_seed
from latsym.tasks import create_dataloaders
from latsym.train import train_loop, create_scheduler, plot_loss_curves
from latsym.eval import plot_regression_surface, plot_run_summary
from latsym.metrics import get_metric, list_metrics

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime


def run_single_field(cfg: DictConfig, scalar_field_fn, device: torch.device,
                     output_dir: Path = None, name: str = None):
    """
    Train on a single field and compute metrics.
    
    Args:
        cfg: Hydra config with data, model, train, metrics sections
        scalar_field_fn: Function (x, y, r) -> scalar field values
        device: torch device
        output_dir: If provided, save all artifacts here. If None, use temp directory.
        name: Run name for logging (optional)
    
    Returns:
        Tuple of (all_metric_values dict, oracles dict, history dict)
    """
    import tempfile
    from contextlib import nullcontext
    
    if name:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
    
    # Set global seed for this run
    run_seed = cfg.experiment.seed
    set_global_seed(run_seed)
    
    # Derive separate seeds for different randomness sources
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    
    # Create generator for reproducible DataLoader shuffling
    shuffle_generator = torch.Generator().manual_seed(data_seed)
    
    train_loader, val_loader, full_dataset = create_dataloaders(
        n_samples=cfg.data.n_samples,
        r_min=cfg.data.r_min,
        r_max=cfg.data.r_max,
        scalar_field_fn=scalar_field_fn,
        train_split=cfg.data.train_split,
        batch_size=cfg.train.batch_size,
        seed=data_seed,
        shuffle_generator=shuffle_generator,
    )
    
    # Set model seed before building for reproducible weight initialization
    set_model_seed(model_seed)
    model = build_model(cfg.model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = create_scheduler(optimizer, cfg.train.total_steps, cfg.train.warmup_steps) if cfg.train.use_scheduler else None
    
    # Use output_dir if provided, otherwise temp directory
    use_temp = output_dir is None
    temp_ctx = tempfile.TemporaryDirectory() if use_temp else nullcontext()
    
    with temp_ctx as temp_dir:
        save_dir = Path(temp_dir) if use_temp else Path(output_dir)
        if not use_temp:
            save_dir.mkdir(parents=True, exist_ok=True)
        if cfg.train.total_steps <= 0:
            print("Skipping training (total_steps=0)")
            history = {
                'step': [],
                'train_loss': [],
                'val_loss': [],
                'val_mae': [],
                'lr': [],
                'steps_per_epoch': len(train_loader),
            }
            if not use_temp:
                torch.save(model.state_dict(), save_dir / 'model.pt')
                with open(save_dir / 'metrics.json', 'w') as f:
                    json.dump(history, f, indent=2)
        else:
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
        
        # Get enabled metrics list
        enabled_metrics = list(cfg.metrics.get("enabled", ["Q"]))
        
        # Compute all enabled metrics using OOP pattern
        all_metric_values = {}
        oracles = {}
        
        for metric_name in enabled_metrics:
            metric_cfg = OmegaConf.to_container(cfg.metrics.get(metric_name, {}), resolve=True)
            metric = get_metric(metric_name, **metric_cfg)
            
            # Compute metric values
            values = metric.compute(model, X, device=device)
            all_metric_values[metric_name] = values
            
            # Compute oracle if metric has one
            oracle = None
            if metric.has_oracle:
                oracle = metric.compute_oracle(X, y, scalar_field_fn, device=device)
                oracles[metric_name] = oracle
            
            # Log values
            metric.log_values(values, oracle, logger=print)
            
            # Plot if saving artifacts
            if not use_temp:
                metric.plot(values, save_dir / f'{metric_name}_vs_layer.png', oracle=oracle, run_name=name)
        
        # Save artifacts if output_dir provided
        if not use_temp:
            # Save metric values
            with open(save_dir / 'metric_values.json', 'w') as f:
                json.dump(all_metric_values, f, indent=2)
            
            # Save loss curves if we trained
            if cfg.train.total_steps > 0:
                plot_loss_curves(history, save_dir / 'loss_curves.png')
            
            # Save regression surface
            plot_regression_surface(model, full_dataset, save_dir / 'regression_surface.png', device)
            
            # Save combined summary plot (Q and MI if enabled)
            if "Q" in enabled_metrics:
                Q_values = all_metric_values.get("Q", {})
                MI_values = all_metric_values.get("MI") if "MI" in enabled_metrics else None
                plot_run_summary(
                    history, Q_values, oracles.get('Q'), model, full_dataset, device,
                    save_dir / 'summary.png', run_name=name,
                    MI_values=MI_values,
                    oracle_MI=oracles.get('MI')
                )
            
            plt.close('all')
        
        return all_metric_values, oracles, history


def run_batch_mode(cfg: DictConfig, output_dir: Path, device: torch.device):
    """
    Run experiments specified in a YAML file.
    
    Each run can specify:
        - name: Run name (used for folder and aggregated files)
        - field: Field config with _target_ for Hydra instantiation
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
        field_cfg = run_spec.get('field', {'_target_': 'latsym.tasks.GaussianRing'})
        
        # Instantiate the field via Hydra
        field_fn = hydra.utils.instantiate(field_cfg)
        
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
        all_metrics, oracles, history = run_single_field(
            run_cfg, field_fn, device, 
            output_dir=run_dir, 
            name=run_name
        )
        
        # Track for aggregation
        summary_src = run_dir / 'summary.png'
        if summary_src.exists():
            q_plot_sources.append((run_name, summary_src))
    
    # Create hard links for summary plots in root directory
    print(f"\n{'='*60}")
    print("Aggregating summary plots to root directory")
    print("="*60)
    
    for run_name, src_path in q_plot_sources:
        dest_path = output_dir / f"{run_name}.png"
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
    # Allow shorthand: python script.py benchmark -> experiment.runs_file=config/runs/benchmark.yaml
    run_name = None
    if len(sys.argv) > 1 and sys.argv[1] and not sys.argv[1].startswith('-') and '=' not in sys.argv[1]:
        if not any('experiment.runs_file' in arg for arg in sys.argv[1:]):
            run_name = sys.argv[1]
            sys.argv[1] = f"experiment.runs_file=config/runs/{run_name}.yaml"
    
    # Extract run name from runs_file if not already extracted
    if not run_name:
        for arg in sys.argv[1:]:
            if 'experiment.runs_file=' in arg:
                runs_file = arg.split('=', 1)[1]
                # Extract name from path like "config/runs/benchmark.yaml" -> "benchmark"
                run_name = Path(runs_file).stem
                break
    
    # Set output directory to runname_date format
    if run_name:
        date_str = datetime.now().strftime('%Y-%m-%d')
        output_dir_override = f"hydra.run.dir=experiments/{date_str}_{run_name}"
        sys.argv.append(output_dir_override)
    
    main()
