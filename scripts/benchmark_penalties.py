#!/usr/bin/env python3
"""
Benchmark script for evaluating different symmetry penalties.

Runs training experiments with various combinations of symmetry penalty types,
lambda values, layers, and learning rates. Results saved to CSV files.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


from latsym.models import build_model
from latsym.seeds import derive_seed, set_model_seed, set_global_seed
from latsym.tasks import create_dataloaders
from latsym.train import train_loop, create_scheduler
from latsym.symmetry_penalty import create_symmetry_penalty, PeriodicPCAOrbitVariancePenalty
from latsym.metrics import get_metric

from omegaconf import OmegaConf, DictConfig


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULTS = {
    "num_hidden_layers": 6,
    "hidden_dim": 128,
    "learning_rate": 1e-4,
    "total_steps": 10000,
    "batch_size": 64,
    "n_samples": 1000,
    "lambda_values": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
    "penalty_types": ["Q_h"],
    "seeds": [42],
    "output_dir": "results",
}


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


def get_completed_runs(csv_path: Path) -> Set[Tuple[float, str]]:
    """
    Load existing CSV and return set of completed (lambda_sym, penalized_layer) tuples.
    
    Args:
        csv_path: Path to CSV file.
    
    Returns:
        Set of (lambda_sym, penalized_layer) tuples for completed runs.
        penalized_layer is stored as string ('' for baseline, '-1' for output, '1' for layer 1, etc.)
    """
    if not csv_path.exists():
        return set()
    
    try:
        df = pd.read_csv(csv_path)
        completed = set()
        for _, row in df.iterrows():
            lambda_sym = row['lambda_sym']
            pen_layer = str(row['penalized_layer']) if pd.notna(row['penalized_layer']) else ''
            completed.add((lambda_sym, pen_layer))
        return completed
    except Exception as e:
        print(f"Warning: Could not read existing CSV {csv_path}: {e}")
        return set()


def run_benchmark(
    num_hidden_layers: int,
    hidden_dim: int,
    run_seeds: List[int],
    learning_rates: List[float],
    lambda_values: List[float],
    penalty_types: List[str],
    sym_layers: List[int],
    total_steps: int,
    output_dir: str,
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
        sym_layers: List of layer indices to penalize.
        total_steps: Total training steps per experiment.
        output_dir: Base directory to save results.
        verbose: Whether to print progress.
    """
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
    
    # Count total experiments per penalty type
    n_nonzero_lambdas = len([l for l in lambda_values if l > 0])
    n_per_penalty = (1 + n_nonzero_lambdas * len(sym_layers)) * len(learning_rates) * len(run_seeds)
    total_experiments = n_per_penalty * len(penalty_types)
    
    print("=" * 60)
    print("Benchmark Configuration")
    print("=" * 60)
    print(f"Model: {num_hidden_layers}x{hidden_dim} MLP")
    print(f"Total steps: {total_steps}")
    print(f"Learning rates: {learning_rates}")
    print(f"Lambda values: {lambda_values}")
    print(f"Penalty types: {penalty_types}")
    print(f"Seeds: {run_seeds}")
    print(f"Layers to penalize: {sym_layers}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")
    
    experiment_count = 0
    
    for penalty_type in penalty_types:
        # Create penalty-specific output directory: results/{penalty}_penalty/runs/
        penalty_output_path = Path(output_dir) / f"{penalty_type}_penalty" / "runs"
        penalty_output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running {penalty_type} penalty experiments")
        print(f"Output: {penalty_output_path}")
        print(f"{'='*60}")
        
        for seed in run_seeds:
            for lr in learning_rates:
                # Create CSV filename for this seed/lr combination
                lr_str = f'{lr:.0e}'.replace('-0', '-')
                csv_filename = penalty_output_path / f'benchmark_layers={num_hidden_layers}x{hidden_dim}_lr={lr_str}_seed={seed}.csv'
                
                # Check which runs are already completed
                completed_runs = get_completed_runs(csv_filename)
                n_expected = 1 + n_nonzero_lambdas * len(sym_layers)
                
                if len(completed_runs) >= n_expected:
                    print(f"Skipping (all {n_expected} runs complete): {csv_filename}")
                    experiment_count += n_expected
                    continue
                elif len(completed_runs) > 0:
                    print(f"\nResuming {csv_filename} ({len(completed_runs)}/{n_expected} runs complete)")
                else:
                    # Create new CSV file with header
                    with open(csv_filename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                    print(f"\nResults will be saved to: {csv_filename}")
                
                # Run baseline (no penalty)
                experiment_count += 1
                baseline_key = (0.0, '')
                if baseline_key in completed_runs:
                    print(f"[{experiment_count}/{total_experiments}] Baseline: seed={seed}, lr={lr} (already complete)")
                else:
                    print(f"\n[{experiment_count}/{total_experiments}] Baseline: seed={seed}, lr={lr}")
                    
                    cfg = create_config(
                        num_hidden_layers=num_hidden_layers,
                        hidden_dim=hidden_dim,
                        learning_rate=lr,
                        total_steps=total_steps,
                        seed=seed,
                        lambda_sym=0.0,
                        sym_layers=[],
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
                
                # Run with symmetry penalties
                for lambda_sym in lambda_values:
                    if lambda_sym == 0.0:
                        continue  # Already ran baseline
                    
                    for penalized_layer in sym_layers:
                        experiment_count += 1
                        layer_str = 'output' if penalized_layer == -1 else str(penalized_layer)
                        run_key = (lambda_sym, str(penalized_layer))
                        
                        if run_key in completed_runs:
                            print(f"[{experiment_count}/{total_experiments}] {penalty_type}: lambda={lambda_sym}, layer={layer_str} (already complete)")
                            continue
                        
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
    print(f"Results saved to: {output_dir}")
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


def parse_layers(layer_spec: str, num_hidden_layers: int) -> List[int]:
    """
    Parse layer spec string into list of layer indices.
    
    Format: compact string where each character specifies a layer:
        - '1'-'9': hidden layer indices
        - 'l' or 'L': output layer (-1)
        - 'a' or 'A': all layers (hidden + output)
    
    Args:
        layer_spec: Layer specification string (e.g., "123l", "l", "a")
        num_hidden_layers: Number of hidden layers in the model.
    
    Returns:
        List of layer indices (1-based for hidden, -1 for output).
    
    Examples:
        "l" -> [-1] (output only)
        "123" -> [1, 2, 3]
        "12l" -> [1, 2, -1]
        "a" or None -> [1, 2, ..., num_hidden_layers, -1]
    """
    if layer_spec is None or layer_spec.lower() == 'a':
        return list(range(1, num_hidden_layers + 1)) + [-1]
    
    layers = []
    for c in layer_spec.lower():
        if c.isdigit():
            layer_idx = int(c)
            if 1 <= layer_idx <= num_hidden_layers:
                layers.append(layer_idx)
            else:
                print(f"Warning: Layer {layer_idx} out of range (1-{num_hidden_layers}). Skipping.")
        elif c == 'l':
            layers.append(-1)
        elif c == 'a':
            return list(range(1, num_hidden_layers + 1)) + [-1]
        else:
            print(f"Warning: Invalid layer spec character '{c}'. Skipping.")
    
    return layers if layers else list(range(1, num_hidden_layers + 1)) + [-1]


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark different symmetry penalties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (Q_h penalty on all layers)
    python scripts/benchmark_penalties.py
    
    # Run with multiple seeds
    python scripts/benchmark_penalties.py --seeds 42-50
    
    # Test specific penalty types
    python scripts/benchmark_penalties.py --penalty-types N_h Q_h
    
    # Custom lambda values
    python scripts/benchmark_penalties.py --lambda-values 0.0 0.1 1.0 10.0 100.0
    
    # Penalize only output layer
    python scripts/benchmark_penalties.py --layers l
    
    # Penalize specific hidden layers (1, 2, 3)
    python scripts/benchmark_penalties.py --layers 123
    
    # Penalize layers 1, 2 and output
    python scripts/benchmark_penalties.py --layers 12l
    
    # Orbit variance loss on output (N_h penalty on output layer)
    python scripts/benchmark_penalties.py --penalty-types N_h --layers l
        """
    )
    
    parser.add_argument('--num-hidden-layers', type=int,
                        help='Number of hidden layers (default: %(default)s)',
                        default=DEFAULTS["num_hidden_layers"])
    parser.add_argument('--hidden-dim', type=int,
                        help='Size of each hidden layer (default: %(default)s)',
                        default=DEFAULTS["hidden_dim"])
    parser.add_argument('--seeds', nargs='+',
                        help='Seeds to run (individual, comma-separated, or ranges like "1-10")')
    parser.add_argument('--learning-rates', nargs='+',
                        help='Learning rates to test (default: %(default)s)',
                        default=None)
    parser.add_argument('--lambda-values', nargs='+',
                        help='Lambda (penalty weight) values to test')
    parser.add_argument('--penalty-types', nargs='+',
                        choices=['N_h', 'N_z', 'Q_h', 'Q_z', 'Q_h_ns', 'Q_z_ns', 'periodic_pca', 'ema_pca'],
                        help='Symmetry penalty types to test')
    parser.add_argument('--layers', type=str, default=None,
                        help='Layer spec: 1-9=hidden layers, l=output, a=all. '
                             'E.g., "l" for output only, "123l" for layers 1,2,3 and output. '
                             'Default: all layers.')
    parser.add_argument('--total-steps', type=int,
                        help='Total training steps per experiment (default: %(default)s)',
                        default=DEFAULTS["total_steps"])
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save results (default: %(default)s)',
                        default=DEFAULTS["output_dir"])
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Parse list arguments with defaults
    run_seeds = parse_seeds(args.seeds) if args.seeds else DEFAULTS["seeds"]
    learning_rates = parse_floats(args.learning_rates) if args.learning_rates else [DEFAULTS["learning_rate"]]
    lambda_values = parse_floats(args.lambda_values) if args.lambda_values else DEFAULTS["lambda_values"]
    penalty_types = args.penalty_types or DEFAULTS["penalty_types"]
    sym_layers = parse_layers(args.layers, args.num_hidden_layers)
    
    # Validate parsed arguments
    if not run_seeds:
        print("Error: No valid seeds specified.")
        sys.exit(1)
    
    if not learning_rates:
        print("Error: No valid learning rates specified.")
        sys.exit(1)
    
    if not lambda_values:
        print("Error: No valid lambda values specified.")
        sys.exit(1)
    
    run_benchmark(
        num_hidden_layers=args.num_hidden_layers,
        hidden_dim=args.hidden_dim,
        run_seeds=run_seeds,
        learning_rates=learning_rates,
        lambda_values=lambda_values,
        penalty_types=penalty_types,
        sym_layers=sym_layers,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
