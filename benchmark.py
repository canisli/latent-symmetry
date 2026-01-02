#!/usr/bin/env python3
"""
Benchmark script for systematic symmetry loss experiments.

Runs training experiments across different lambda values, layers, learning rates, and seeds
by calling train.py via subprocess with Hydra overrides.

Usage:
    python benchmark.py --seeds 42
    python benchmark.py --seeds 42,43,44
    python benchmark.py --seeds 42-50
    python benchmark.py --num-phi-layers 4 --num-rho-layers 4 --seeds 42
"""

import subprocess
import csv
import argparse
import os
import re
import sys
import time


def row_exists(csv_filename: str, learning_rate: float, lambda_sym_max: float, symmetry_layer: int) -> bool:
    """
    Check if a specific experiment row already exists in the CSV file.
    
    Args:
        csv_filename: Path to CSV file
        learning_rate: Learning rate value
        lambda_sym_max: Lambda symmetry max value
        symmetry_layer: Symmetry layer index (None for baseline)
    
    Returns:
        True if row exists, False otherwise
    """
    if not os.path.exists(csv_filename):
        return False
    
    try:
        with open(csv_filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_lr = float(row['learning_rate'])
                row_lambda = float(row['lambda_sym_max'])
                row_layer_str = row['symmetry_layer']
                
                # Handle symmetry_layer comparison (empty string means None)
                row_layer = None if row_layer_str == '' else int(row_layer_str)
                
                if (abs(row_lr - learning_rate) < 1e-10 and
                    abs(row_lambda - lambda_sym_max) < 1e-10 and
                    row_layer == symmetry_layer):
                    return True
    except (FileNotFoundError, KeyError, ValueError):
        # If file is corrupted or missing columns, treat as not existing
        return False
    
    return False


def parse_training_output(stdout: str) -> dict:
    """
    Parse training output to extract results.
    
    Args:
        stdout: Standard output from train.py
        
    Returns:
        Dictionary with test_task_loss, test_rel_rmse, test_sym_loss
    """
    result = {
        'test_task_loss': None,
        'test_rel_rmse': None,
        'test_sym_loss': None,
    }
    
    for line in stdout.split('\n'):
        if 'Test task loss:' in line:
            match = re.search(r'Test task loss:\s+([\d.e+-]+)', line)
            if match:
                result['test_task_loss'] = float(match.group(1))
        elif 'Test relative RMSE:' in line:
            match = re.search(r'Test relative RMSE:\s+([\d.e+-]+)', line)
            if match:
                result['test_rel_rmse'] = float(match.group(1))
        elif 'Test symmetry loss:' in line:
            match = re.search(r'Test symmetry loss:\s+([\d.e+-]+)', line)
            if match:
                result['test_sym_loss'] = float(match.group(1))
    
    return result


def run_experiment(
    symmetry_layer: int,
    lambda_sym_max: float,
    learning_rate: float,
    num_phi_layers: int,
    num_rho_layers: int,
    hidden_channels: int,
    run_seed: int,
    csv_filename: str,
    fieldnames: list,
    python_cmd: str = 'python',
):
    """
    Run a single training experiment and save results.
    
    Args:
        symmetry_layer: Layer index for symmetry loss (-1 for output, None to disable)
        lambda_sym_max: Maximum lambda for symmetry loss
        learning_rate: Learning rate
        num_phi_layers: Number of phi layers
        num_rho_layers: Number of rho layers
        hidden_channels: Hidden channel dimension
        run_seed: Random seed
        csv_filename: Path to CSV file for results
        fieldnames: CSV column names
        python_cmd: Python command to use
    
    Returns:
        result_dict: Dictionary with experiment results, or None if skipped
    """
    # Check if this experiment already exists
    if row_exists(csv_filename, learning_rate, lambda_sym_max, symmetry_layer):
        layer_str = "None" if symmetry_layer is None else str(symmetry_layer)
        print(f"Skipping: lambda_sym_max={lambda_sym_max}, symmetry_layer={layer_str} (already exists)")
        return None
    
    layer_str = "None" if symmetry_layer is None else str(symmetry_layer)
    print(f"Running: lambda_sym_max={lambda_sym_max}, symmetry_layer={layer_str}")
    
    # Record start time
    start_time = time.time()
    
    # Build command with Hydra overrides
    cmd = [
        python_cmd, 'train.py',
        'headless=true',
        f'run_seed={run_seed}',
        f'training.learning_rate={learning_rate}',
        f'model.num_phi_layers={num_phi_layers}',
        f'model.num_rho_layers={num_rho_layers}',
        f'model.hidden_channels={hidden_channels}',
    ]
    
    if symmetry_layer is not None:
        cmd.extend([
            'symmetry.enabled=true',
            f'symmetry.layer_idx={symmetry_layer}',
            f'symmetry.lambda_sym_max={lambda_sym_max}',
        ])
    else:
        cmd.append('symmetry.enabled=false')
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
        )
        
        if result.returncode != 0:
            print(f"\033[31m  -> Error running experiment:\033[0m")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return None
        
        # Parse output
        parsed = parse_training_output(result.stdout)
        
    except Exception as e:
        print(f"\033[31m  -> Exception: {e}\033[0m")
        return None
    
    result_dict = {
        'learning_rate': learning_rate,
        'lambda_sym_max': lambda_sym_max,
        'symmetry_layer': symmetry_layer,
        'test_task_loss': parsed['test_task_loss'],
        'test_sym_loss': parsed['test_sym_loss'],
        'test_rel_rmse': parsed['test_rel_rmse'],
    }
    
    # Write result immediately to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'learning_rate': result_dict['learning_rate'],
            'lambda_sym_max': result_dict['lambda_sym_max'],
            'symmetry_layer': '' if result_dict['symmetry_layer'] is None else result_dict['symmetry_layer'],
            'test_task_loss': result_dict['test_task_loss'],
            'test_sym_loss': result_dict['test_sym_loss'] if result_dict['test_sym_loss'] is not None else '',
            'test_rel_rmse': result_dict['test_rel_rmse'],
        })
    
    # Calculate duration
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"
    
    # Print results
    sym_loss_str = f"{result_dict['test_sym_loss']:.4e}" if result_dict['test_sym_loss'] is not None else "N/A"
    task_loss_str = f"{result_dict['test_task_loss']:.4e}" if result_dict['test_task_loss'] is not None else "N/A"
    rel_rmse_str = f"{result_dict['test_rel_rmse']:.4f}" if result_dict['test_rel_rmse'] is not None else "N/A"
    print(f"\033[32m  -> Task loss: {task_loss_str}, Sym loss: {sym_loss_str}, Rel RMSE: {rel_rmse_str}\033[0m")
    print(f"\033[32m  -> Time: {duration_str}\033[0m")
    print(f"\033[32m  -> Saved to {csv_filename}\033[0m\n")
    
    return result_dict


def run_benchmark(
    num_phi_layers: int = 3,
    num_rho_layers: int = 3,
    hidden_channels: int = 128,
    run_seeds: list = None,
    python_cmd: str = 'python',
):
    """
    Run training experiments with different lambda_sym and symmetry_layer values.
    
    Args:
        num_phi_layers: Number of phi layers in DeepSets
        num_rho_layers: Number of rho layers in DeepSets
        hidden_channels: Hidden channel dimension
        run_seeds: List of seeds to run
        python_cmd: Python command to use
    """
    if run_seeds is None:
        run_seeds = [42]
    
    # Experiment configuration
    #lambda_sym_values = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    lambda_sym_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    # Compute valid symmetry layers based on architecture
    # Layers 1..num_phi_layers: phi activations
    # Layer num_phi_layers + 1: post-pooling
    # Layers num_phi_layers + 2..num_phi_layers + num_rho_layers + 1: rho activations
    max_layer = num_phi_layers + num_rho_layers + 1
    symmetry_layers = list(range(1, max_layer + 1)) + [-1]
    
    learning_rates = [1e-4] # [1e-4, 3e-4, 6e-4]
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    fieldnames = ['learning_rate', 'lambda_sym_max', 'symmetry_layer', 
                  'test_task_loss', 'test_sym_loss', 'test_rel_rmse']
    
    print("Starting benchmark experiments...")
    print(f"Lambda_sym values: {lambda_sym_values}")
    print(f"Symmetry layers: {symmetry_layers} + None")
    print(f"Learning rates: {learning_rates}")
    print(f"Seeds: {run_seeds}")
    print(f"Architecture: num_phi_layers={num_phi_layers}, num_rho_layers={num_rho_layers}")
    total_per_seed = len(lambda_sym_values) * len(symmetry_layers) + 1  # +1 for None case
    print(f"Total experiments per seed per lr: {total_per_seed}")
    print(f"Total experiments: {len(run_seeds) * len(learning_rates) * total_per_seed}\n")
    
    # Run for each seed
    for seed_idx, run_seed in enumerate(run_seeds):
        for learning_rate in learning_rates:
            lr_string = f'{learning_rate:.0e}'.replace('-0', '-')
            csv_filename = f'results/model=deepsets_layers={num_phi_layers}+{num_rho_layers}_lr={lr_string}_seed={run_seed}.csv'
            
            print(f"\n{'='*60}")
            print(f"Running seed {run_seed} ({seed_idx + 1}/{len(run_seeds)}), lr={learning_rate:.0e}")
            print(f"{'='*60}")
            print(f"Results will be saved incrementally to {csv_filename}\n")
            
            # Initialize CSV file with header if it doesn't exist
            if not os.path.exists(csv_filename):
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
            
            # Run experiments with symmetry layers
            for lambda_sym_max in lambda_sym_values:
                for symmetry_layer in symmetry_layers:
                    run_experiment(
                        symmetry_layer, lambda_sym_max, learning_rate,
                        num_phi_layers, num_rho_layers, hidden_channels,
                        run_seed, csv_filename, fieldnames, python_cmd
                    )
            
            # Run baseline experiment with symmetry_layer=None
            run_experiment(
                None, 0.0, learning_rate,
                num_phi_layers, num_rho_layers, hidden_channels,
                run_seed, csv_filename, fieldnames, python_cmd
            )
            
            print(f"\nSeed {run_seed}, lr={learning_rate:.0e} complete. Results saved to {csv_filename}")
    
    print(f"\n{'='*60}")
    print(f"All benchmarks complete! Processed {len(run_seeds)} seed(s).")
    print(f"{'='*60}")


def parse_seeds(seed_args: list) -> list:
    """
    Parse seed arguments that can include individual seeds, comma-separated seeds, or ranges.
    
    Args:
        seed_args: List of seed arguments (e.g., ['42'], ['42,43,44'], ['42-50'])
    
    Returns:
        List of seed integers
    """
    seeds = []
    
    for arg in seed_args:
        # Check if it's a range (contains '-')
        if '-' in arg and not arg.startswith('-'):
            try:
                start, end = arg.split('-')
                start = int(start.strip())
                end = int(end.strip())
                if start > end:
                    print(f"Warning: Invalid range '{arg}' (start > end). Skipping.")
                    continue
                seeds.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{arg}'. Skipping.")
                continue
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
    
    # Remove duplicates while preserving order
    seen = set()
    unique_seeds = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            unique_seeds.append(seed)
    
    return sorted(unique_seeds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run benchmark experiments with different hyperparameters'
    )
    parser.add_argument('--num-phi-layers', type=int, default=3,
                        help='Number of phi layers (default: 3)')
    parser.add_argument('--num-rho-layers', type=int, default=3,
                        help='Number of rho layers (default: 3)')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Hidden channel dimension (default: 128)')
    parser.add_argument('--seeds', nargs='+', default=['42'],
                        help='Seeds to run (individual, comma-separated, or ranges like "42-50")')
    parser.add_argument('--python', type=str, default='python',
                        help='Python command to use (default: python)')
    
    args = parser.parse_args()
    
    # Parse seeds
    run_seeds = parse_seeds(args.seeds)
    
    if not run_seeds:
        print("Error: No valid seeds specified.")
        sys.exit(1)
    
    run_benchmark(
        num_phi_layers=args.num_phi_layers,
        num_rho_layers=args.num_rho_layers,
        hidden_channels=args.hidden_channels,
        run_seeds=run_seeds,
        python_cmd=args.python,
    )

