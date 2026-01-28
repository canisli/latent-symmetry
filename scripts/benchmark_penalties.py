#!/usr/bin/env python3
"""
Benchmark script for evaluating different symmetry penalties.

This script runs training experiments with various combinations of:
- Symmetry penalty types (N_h, N_z, Q_h, Q_z, etc.)
- Lambda values (penalty strength)
- Layers to penalize
- Learning rates

Results are saved incrementally to CSV files for analysis.

Usage:
    python scripts/benchmark_penalties.py --seeds 42 43 44
    python scripts/benchmark_penalties.py --seeds 1-10 --num-hidden-layers 6 --hidden-dim 128
    python scripts/benchmark_penalties.py --penalty-types N_h Q_h --lambda-values 0.1 1.0 10.0
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np


from latsym.models import build_model
from latsym.seeds import derive_seed, set_model_seed, set_global_seed
from latsym.tasks import create_dataloaders
from latsym.train import train_loop, create_scheduler
from latsym.symmetry_penalty import create_symmetry_penalty, PeriodicPCAOrbitVariancePenalty
from latsym.metrics import get_metric

from omegaconf import OmegaConf, DictConfig


def create_config(
    num_hidden_layers: int = 6,
    hidden_dim: int = 128,
    learning_rate: float = 1e-4,
    total_steps: int = 10000,
    batch_size: int = 64,
    n_samples: int = 1000,
    seed: int = 42,
    lambda_sym: float = 0.0,
    sym_layers: List[int] = None,
    sym_penalty_type: str = "Q_z",
    weight_decay: float = 0.0,
    warmup_steps: int = 100,
    use_scheduler: bool = True,
    n_augmentations_train: int = 4,
    stopgrad_denominator: bool = True,
) -> DictConfig:
    """
    Create a Hydra-compatible configuration dictionary.
    
    Args:
        num_hidden_layers: Number of hidden layers in the MLP.
        hidden_dim: Size of each hidden layer.
        learning_rate: Learning rate for optimizer.
        total_steps: Total training steps.
        batch_size: Batch size for training.
        n_samples: Number of data samples.
        seed: Random seed for reproducibility.
        lambda_sym: Weight for symmetry penalty (0 = disabled).
        sym_layers: List of layer indices to penalize.
        sym_penalty_type: Type of symmetry penalty.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for scheduler.
        use_scheduler: Whether to use learning rate scheduler.
        n_augmentations_train: Number of rotation pairs per sample.
        stopgrad_denominator: Stop gradient through denominator in Q penalties.
    
    Returns:
        OmegaConf configuration dictionary.
    """
    if sym_layers is None:
        sym_layers = []
    
    cfg = {
        "experiment": {
            "name": "benchmark",
            "seed": seed,
        },
        "data": {
            "n_samples": n_samples,
            "r_min": 0.1,
            "r_max": 2.0,
            "train_split": 0.8,
        },
        "model": {
            "input_dim": 2,
            "hidden_dim": hidden_dim,
            "num_layers": num_hidden_layers,
            "output_dim": 1,
            "activation": "relu",
        },
        "train": {
            "batch_size": batch_size,
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "use_scheduler": use_scheduler,
            "warmup_steps": warmup_steps,
            "log_interval": 100,
            "eval_interval": 500,
            "save_best": False,  # Don't save models in benchmark mode
            "lambda_sym": lambda_sym,
            "sym_layers": sym_layers,
            "sym_penalty_type": sym_penalty_type,
            "stopgrad_denominator": stopgrad_denominator,
            "n_augmentations_train": n_augmentations_train,
            "dynamics_mode": False,
            "grad_align_interval": 0,
        },
        "metrics": {
            "Q": {"n_rotations": 32, "explained_variance": 0.95},
            "Q_h": {"n_rotations": 32},
            "RSL": {"n_rotations": 32, "epsilon": 1e-8},
            "SL": {"n_rotations": 32},
        },
    }
    
    return OmegaConf.create(cfg)


def run_single_experiment(
    cfg: DictConfig,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single training experiment with the given configuration.
    
    Args:
        cfg: Hydra configuration.
        output_dir: Optional directory to save outputs.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with experiment results including metrics.
    """
    # Set global seed
    run_seed = cfg.experiment.seed
    set_global_seed(run_seed)
    
    # Derive separate seeds
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    augmentation_seed = derive_seed(run_seed, "augmentation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create generators for reproducible randomness
    shuffle_generator = torch.Generator().manual_seed(data_seed)
    augmentation_generator = torch.Generator(device=device).manual_seed(augmentation_seed)
    
    # Create data loaders
    train_loader, val_loader, full_dataset = create_dataloaders(
        n_samples=cfg.data.n_samples,
        r_min=cfg.data.r_min,
        r_max=cfg.data.r_max,
        train_split=cfg.data.train_split,
        batch_size=cfg.train.batch_size,
        seed=data_seed,
        shuffle_generator=shuffle_generator,
    )
    
    # Build model
    set_model_seed(model_seed)
    model = build_model(cfg.model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay
    )
    scheduler = create_scheduler(
        optimizer, cfg.train.total_steps, cfg.train.warmup_steps
    ) if cfg.train.use_scheduler else None
    
    # Setup symmetry penalty
    lambda_sym = cfg.train.get('lambda_sym', 0.0)
    sym_layers = list(cfg.train.get('sym_layers', []))
    sym_penalty_type = cfg.train.get('sym_penalty_type', 'N_h')
    n_augmentations = cfg.train.get('n_augmentations_train', 4)
    stopgrad_denominator = cfg.train.get('stopgrad_denominator', True)
    
    symmetry_penalty = None
    if lambda_sym > 0 and sym_layers:
        symmetry_penalty = create_symmetry_penalty(
            sym_penalty_type,
            stopgrad_denominator=stopgrad_denominator
        )
        
        # For periodic PCA penalty, set reference data
        if isinstance(symmetry_penalty, PeriodicPCAOrbitVariancePenalty):
            X_train = torch.cat([batch[0] for batch in train_loader], dim=0)
            symmetry_penalty.set_reference_data(X_train)
    
    # Training loop
    history = train_loop(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, device,
        cfg.train.total_steps, cfg.train.log_interval, cfg.train.eval_interval,
        output_dir, cfg.train.save_best,
        symmetry_penalty=symmetry_penalty,
        lambda_sym=lambda_sym,
        sym_layers=sym_layers,
        n_augmentations=n_augmentations,
        augmentation_generator=augmentation_generator,
    )
    
    # Compute metrics on the trained model
    X, y = full_dataset.get_numpy()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)
    scalar_field_fn = full_dataset.scalar_field_fn
    
    # Compute Q metric (PCA-projected orbit variance ratio)
    q_cfg = OmegaConf.to_container(cfg.metrics.get("Q", {}), resolve=True)
    q_metric = get_metric("Q", **q_cfg)
    Q_values = q_metric.compute(model, X_tensor, device=device)
    oracle_Q = q_metric.compute_oracle(X_tensor, y_tensor, scalar_field_fn, device=device)
    
    # Compute Q_h metric (raw activation orbit variance ratio)
    qh_cfg = OmegaConf.to_container(cfg.metrics.get("Q_h", {}), resolve=True)
    qh_metric = get_metric("Q_h", **qh_cfg)
    Q_h_values = qh_metric.compute(model, X_tensor, device=device)
    
    # Extract final metrics
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else None
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
    final_val_mae = history['val_mae'][-1] if history['val_mae'] else None
    
    # Best validation loss during training
    best_val_loss = min(history['val_loss']) if history['val_loss'] else None
    
    result = {
        # Configuration
        'seed': run_seed,
        'learning_rate': cfg.train.learning_rate,
        'lambda_sym': lambda_sym,
        'sym_penalty_type': sym_penalty_type if lambda_sym > 0 and sym_layers else 'none',
        'penalized_layer': sym_layers[0] if sym_layers else '',
        'num_hidden_layers': cfg.model.num_layers,
        'hidden_dim': cfg.model.hidden_dim,
        
        # Training metrics
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'final_val_mae': final_val_mae,
        
        # Oracle Q value
        'oracle_Q': oracle_Q,
    }
    
    # Add per-layer Q values (flatten from dict)
    # Note: metric returns 'layer_1', 'layer_2', ..., 'output' keys
    for key, value in Q_values.items():
        if key == 'output':
            result['Q_layer_-1'] = value
        else:
            result[f'Q_{key}'] = value
    
    # Add per-layer Q_h values (flatten from dict)
    for key, value in Q_h_values.items():
        if key == 'output':
            result['Q_h_layer_-1'] = value
        else:
            result[f'Q_h_{key}'] = value
    
    return result


def run_benchmark(
    num_hidden_layers: int = 6,
    hidden_dim: int = 128,
    run_seeds: List[int] = None,
    learning_rates: List[float] = None,
    lambda_values: List[float] = None,
    penalty_types: List[str] = None,
    total_steps: int = 10000,
    output_dir: str = "results/benchmark",
    verbose: bool = True,
):
    """
    Run benchmark experiments with different hyperparameter combinations.
    
    Args:
        num_hidden_layers: Number of hidden layers.
        hidden_dim: Size of each hidden layer.
        run_seeds: List of random seeds to run.
        learning_rates: List of learning rates to try.
        lambda_values: List of lambda (penalty weight) values.
        penalty_types: List of penalty types to test.
        total_steps: Total training steps per experiment.
        output_dir: Directory to save results.
        verbose: Whether to print progress.
    """
    if run_seeds is None:
        run_seeds = [42]
    if learning_rates is None:
        learning_rates = [1e-4]
    if lambda_values is None:
        lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    if penalty_types is None:
        penalty_types = ["Q_h"]
    
    # Layers to penalize: all hidden layers and output
    all_sym_layers = list(range(1, num_hidden_layers + 1)) + [-1]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # CSV fieldnames - base fields plus per-layer Q and Q_h values
    base_fieldnames = [
        'seed', 'learning_rate', 'lambda_sym', 'sym_penalty_type', 'penalized_layer',
        'num_hidden_layers', 'hidden_dim',
        'final_train_loss', 'final_val_loss', 'best_val_loss', 'final_val_mae',
        'oracle_Q',
    ]
    # Per-layer fields: Q_layer_1, Q_layer_2, ..., Q_layer_{n}, Q_layer_-1 (output)
    q_fieldnames = [f'Q_layer_{i}' for i in range(1, num_hidden_layers + 1)] + ['Q_layer_-1']
    qh_fieldnames = [f'Q_h_layer_{i}' for i in range(1, num_hidden_layers + 1)] + ['Q_h_layer_-1']
    fieldnames = base_fieldnames + q_fieldnames + qh_fieldnames
    
    # Count total experiments
    # For lambda=0, we only run once (no penalty), for lambda>0 we test each penalty type × each layer
    n_baseline = len(learning_rates) * len(run_seeds)
    n_nonzero_lambdas = len([l for l in lambda_values if l > 0])
    n_with_penalty = n_nonzero_lambdas * len(penalty_types) * len(all_sym_layers) * len(learning_rates) * len(run_seeds)
    total_experiments = n_baseline + n_with_penalty
    
    print("=" * 60)
    print("Benchmark Configuration")
    print("=" * 60)
    print(f"Model: {num_hidden_layers}x{hidden_dim} MLP")
    print(f"Total steps: {total_steps}")
    print(f"Learning rates: {learning_rates}")
    print(f"Lambda values: {lambda_values}")
    print(f"Penalty types: {penalty_types}")
    print(f"Seeds: {run_seeds}")
    print(f"Layers to penalize: {all_sym_layers}")
    print(f"Total experiments: {total_experiments} (1 baseline + {n_nonzero_lambdas} lambdas × {len(penalty_types)} penalties × {len(all_sym_layers)} layers per seed/lr)")
    print("=" * 60 + "\n")
    
    experiment_count = 0
    
    for seed in run_seeds:
        for lr in learning_rates:
            # Create CSV filename for this seed/lr combination
            lr_str = f'{lr:.0e}'.replace('-0', '-')
            csv_filename = output_path / f'benchmark_layers={num_hidden_layers}x{hidden_dim}_lr={lr_str}_seed={seed}.csv'
            
            # Check if file exists and skip if so
            if csv_filename.exists():
                print(f"Skipping (file exists): {csv_filename}")
                # Count how many experiments we're skipping
                n_skip = 1 + len([l for l in lambda_values if l > 0]) * len(penalty_types) * len(all_sym_layers)
                experiment_count += n_skip
                continue
            
            # Open CSV file and write header
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            print(f"\nResults will be saved to: {csv_filename}")
            
            # Run baseline (no penalty)
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] Baseline: seed={seed}, lr={lr}")
            
            cfg = create_config(
                num_hidden_layers=num_hidden_layers,
                hidden_dim=hidden_dim,
                learning_rate=lr,
                total_steps=total_steps,
                seed=seed,
                lambda_sym=0.0,
                sym_layers=[],
                sym_penalty_type="N_h",
            )
            
            result = run_single_experiment(cfg, verbose=verbose)
            
            # Write result to CSV
            row = {k: result.get(k, '') for k in fieldnames}
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
            
            # Compute average Q for display
            q_vals = [result.get(f'Q_layer_{i}', 0) for i in range(1, num_hidden_layers + 1)] + [result.get('Q_layer_-1', 0)]
            avg_q = np.mean([v for v in q_vals if v is not None and v != ''])
            print(f"  -> Val loss: {result['final_val_loss']:.4e}, Avg Q: {avg_q:.4f}")
            
            # Run with symmetry penalties
            for lambda_sym in lambda_values:
                if lambda_sym == 0.0:
                    continue  # Already ran baseline
                
                for penalty_type in penalty_types:
                    for penalized_layer in all_sym_layers:
                        experiment_count += 1
                        layer_str = 'output' if penalized_layer == -1 else str(penalized_layer)
                        print(f"\n[{experiment_count}/{total_experiments}] {penalty_type}: lambda={lambda_sym}, layer={layer_str}, seed={seed}, lr={lr}")
                        
                        cfg = create_config(
                            num_hidden_layers=num_hidden_layers,
                            hidden_dim=hidden_dim,
                            learning_rate=lr,
                            total_steps=total_steps,
                            seed=seed,
                            lambda_sym=lambda_sym,
                            sym_layers=[penalized_layer],
                            sym_penalty_type=penalty_type,
                        )
                        
                        result = run_single_experiment(cfg, verbose=verbose)
                        
                        # Write result to CSV
                        row = {k: result.get(k, '') for k in fieldnames}
                        with open(csv_filename, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow(row)
                        
                        # Compute average Q for display
                        q_vals = [result.get(f'Q_layer_{i}', 0) for i in range(1, num_hidden_layers + 1)] + [result.get('Q_layer_-1', 0)]
                        avg_q = np.mean([v for v in q_vals if v is not None and v != ''])
                        print(f"  -> Val loss: {result['final_val_loss']:.4e}, Avg Q: {avg_q:.4f}")
            
            print(f"\nSaved results to: {csv_filename}")
    
    print("\n" + "=" * 60)
    print(f"Benchmark complete! Ran {experiment_count} experiments.")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


def parse_seeds(seed_args: List[str]) -> List[int]:
    """
    Parse seed arguments that can include individual seeds, comma-separated seeds, or ranges.
    
    Args:
        seed_args: List of seed arguments (e.g., ["42", "1-5", "10,11,12"])
    
    Returns:
        List of unique sorted seed integers.
    """
    seeds = []
    
    for arg in seed_args:
        # Check if it's a range (contains '-')
        if '-' in arg and not arg.startswith('-'):
            try:
                start, end = arg.split('-')
                start = int(start.strip())
                end = int(end.strip())
                if start <= end:
                    seeds.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{arg}'. Skipping.")
        # Check if it's comma-separated
        elif ',' in arg:
            for s in arg.split(','):
                try:
                    seeds.append(int(s.strip()))
                except ValueError:
                    print(f"Warning: Invalid seed '{s}'. Skipping.")
        # Single seed
        else:
            try:
                seeds.append(int(arg))
            except ValueError:
                print(f"Warning: Invalid seed '{arg}'. Skipping.")
    
    # Remove duplicates and sort
    return sorted(set(seeds))


def parse_floats(float_args: List[str]) -> List[float]:
    """Parse a list of string arguments into floats."""
    result = []
    for arg in float_args:
        if ',' in arg:
            for f in arg.split(','):
                try:
                    result.append(float(f.strip()))
                except ValueError:
                    print(f"Warning: Invalid float '{f}'. Skipping.")
        else:
            try:
                result.append(float(arg))
            except ValueError:
                print(f"Warning: Invalid float '{arg}'. Skipping.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark different symmetry penalties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python scripts/benchmark_penalties.py
    
    # Run with multiple seeds
    python scripts/benchmark_penalties.py --seeds 42 43 44
    python scripts/benchmark_penalties.py --seeds 1-10
    
    # Test specific penalty types
    python scripts/benchmark_penalties.py --penalty-types N_h Q_h
    
    # Custom lambda values
    python scripts/benchmark_penalties.py --lambda-values 0.0 0.1 1.0 10.0 100.0
    
    # Custom model architecture
    python scripts/benchmark_penalties.py --num-hidden-layers 8 --hidden-dim 256
        """
    )
    
    parser.add_argument('--num-hidden-layers', type=int,
                        help='Number of hidden layers (uses default if omitted)')
    parser.add_argument('--hidden-dim', type=int,
                        help='Size of each hidden layer (uses default if omitted)')
    parser.add_argument('--seeds', nargs='+',
                        help='Seeds to run (individual, comma-separated, or ranges like "1-10")')
    parser.add_argument('--learning-rates', nargs='+',
                        help='Learning rates to test (uses default if omitted)')
    parser.add_argument('--lambda-values', nargs='+',
                        help='Lambda (penalty weight) values to test (uses default if omitted)')
    parser.add_argument('--penalty-types', nargs='+',
                        choices=['N_h', 'N_z', 'Q_h', 'Q_z', 'periodic_pca', 'ema_pca'],
                        help='Symmetry penalty types to test (uses default if omitted)')
    parser.add_argument('--total-steps', type=int,
                        help='Total training steps per experiment (uses default if omitted)')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save results (uses default if omitted)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Parse arguments
    run_seeds = parse_seeds(args.seeds) if args.seeds is not None else None
    learning_rates = parse_floats(args.learning_rates) if args.learning_rates is not None else None
    lambda_values = parse_floats(args.lambda_values) if args.lambda_values is not None else None
    
    if run_seeds is not None and not run_seeds:
        print("Error: No valid seeds specified.")
        sys.exit(1)
    
    if learning_rates is not None and not learning_rates:
        print("Error: No valid learning rates specified.")
        sys.exit(1)
    
    if lambda_values is not None and not lambda_values:
        print("Error: No valid lambda values specified.")
        sys.exit(1)
    
    run_kwargs = {
        "run_seeds": run_seeds,
        "learning_rates": learning_rates,
        "lambda_values": lambda_values,
        "penalty_types": args.penalty_types,
        "verbose": not args.quiet,
    }
    if args.num_hidden_layers is not None:
        run_kwargs["num_hidden_layers"] = args.num_hidden_layers
    if args.hidden_dim is not None:
        run_kwargs["hidden_dim"] = args.hidden_dim
    if args.total_steps is not None:
        run_kwargs["total_steps"] = args.total_steps
    if args.output_dir is not None:
        run_kwargs["output_dir"] = args.output_dir
    
    run_benchmark(**run_kwargs)


if __name__ == '__main__':
    main()
