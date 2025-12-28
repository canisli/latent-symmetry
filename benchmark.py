import train
import csv
import argparse
import os

def run_experiment(symmetry_layer, lambda_sym_max, learning_rate, num_hidden_layers, hidden_dim, run_seed, csv_filename, fieldnames):
    """
    Run a single training experiment and save results.
    
    Returns:
        result_dict: Dictionary with experiment results
    """
    layer_str = "None" if symmetry_layer is None else str(symmetry_layer)
    print(f"Running: lambda_sym_max={lambda_sym_max}, symmetry_layer={layer_str}")
    
    result = train.main(
        headless=True,
        symmetry_layer=symmetry_layer,
        lambda_sym_max=lambda_sym_max,
        learning_rate=learning_rate,
        num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        run_seed=run_seed
    )
    
    result_dict = {
        'learning_rate': result['learning_rate'],
        'lambda_sym_max': result['lambda_sym_max'],
        'symmetry_layer': result['symmetry_layer'],
        'test_task_loss': result['test_task_loss'],
        'test_sym_loss': result['test_sym_loss'] if result['test_sym_loss'] is not None else 0.0
    }
    
    # Write result immediately to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'learning_rate': result_dict['learning_rate'],
            'lambda_sym_max': result_dict['lambda_sym_max'],
            'symmetry_layer': '' if result_dict['symmetry_layer'] is None else result_dict['symmetry_layer'],
            'test_task_loss': result_dict['test_task_loss'],
            'test_sym_loss': result_dict['test_sym_loss'] if result_dict['test_sym_loss'] is not None else ''
        })
    
    # Print results
    sym_loss_str = f"{result_dict['test_sym_loss']:.4e}" if result_dict['test_sym_loss'] is not None else "N/A"
    print(f"\033[32m  -> Task loss: {result_dict['test_task_loss']:.4e}, Sym loss: {sym_loss_str}\033[0m")
    print(f"\033[32m  -> Saved to {csv_filename}\033[0m\n")
    
    return result_dict

def run_benchmark(num_hidden_layers=6, hidden_dim=128, run_seeds=[42]):
    """
    Run training experiments with different lambda_sym and symmetry_layer values.
    Can run with multiple seeds, saving each to a separate CSV file.
    
    Args:
        num_hidden_layers: Number of hidden layers
        hidden_dim: Size of each hidden layer
        run_seeds: List of seeds to run (each seed gets its own CSV file)
    """
    # Experiment configuration
    # lambda_sym_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 100.0]
    lambda_sym_values = [
        0,
        0.001, 0.003,
        0.01, 0.03,
        0.1, 0.2, 0.3, 0.5,
        1.0, 2.0, 3.0, 5.0,
        10.0, 100.0
    ]

    symmetry_layers = list(range(1, num_hidden_layers + 1)) + [-1]  # Layers 1 to num_hidden_layers, plus -1
    learning_rates = [1e-4, 3e-4, 6e-4]
    
    
    print("Starting benchmark experiments...")
    print(f"Lambda_sym values: {lambda_sym_values}")
    print(f"Symmetry layers: {symmetry_layers} + None")
    print(f"Learning rates: {learning_rates}")
    print(f"Seeds: {run_seeds}")
    print(f"Total experiments per seed: {len(lambda_sym_values) * len(symmetry_layers) + 1}")  # +1 for None case
    print(f"Total experiments: {len(run_seeds) * (len(lambda_sym_values) * len(symmetry_layers) + 1)}\n")
    
    # Run for each seed
    for seed_idx, run_seed in enumerate(run_seeds):
        for learning_rate in learning_rates:
            lr_string = f'{learning_rate:.0e}'.replace('-0','-')
            csv_filename = f'results/layers={num_hidden_layers}x{hidden_dim}_lr={lr_string}_seed={run_seed}.csv'
            fieldnames = ['learning_rate', 'lambda_sym_max', 'symmetry_layer', 'test_task_loss', 'test_sym_loss']
            
            # Skip if file already exists
            if os.path.exists(csv_filename):
                print(f"\n{'='*60}")
                print(f"Skipping seed {run_seed}, lr={learning_rate:.0e} (file already exists: {csv_filename})")
                print(f"{'='*60}\n")
                continue
            
            print(f"\n{'='*60}")
            print(f"Running seed {run_seed} ({seed_idx + 1}/{len(run_seeds)})")
            print(f"{'='*60}")
            print(f"Results will be saved incrementally to {csv_filename}\n")
            
            # Open CSV file and write header
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            # Run experiments with symmetry layers
            for lambda_sym_max in lambda_sym_values:
                for symmetry_layer in symmetry_layers:
                    run_experiment(symmetry_layer, lambda_sym_max, learning_rate, num_hidden_layers, hidden_dim, run_seed, csv_filename, fieldnames)
            
            # Run baseline experiment with symmetry_layer=None
            run_experiment(None, 0.0, learning_rate, num_hidden_layers, hidden_dim, run_seed, csv_filename, fieldnames)

            print(f"\nSeed {run_seed} complete. Results saved to {csv_filename}")
    
    print(f"\n{'='*60}")
    print(f"All benchmarks complete! Processed {len(run_seeds)} seed(s).")
    print(f"{'='*60}")


def parse_seeds(seed_args):
    """
    Parse seed arguments that can include individual seeds, comma-separated seeds, or ranges.
    
    Args:
        seed_args: List of seed arguments (can be individual seeds, comma-separated, or ranges like "63-100")
    
    Returns:
        List of seed integers
    """
    seeds = []
    
    for arg in seed_args:
        # Check if it's a range (contains '-')
        if '-' in arg:
            try:
                start, end = arg.split('-')
                start = int(start.strip())
                end = int(end.strip())
                if start > end:
                    print(f"Warning: Invalid range '{arg}' (start > end). Skipping.")
                    continue
                seeds.extend(range(start, end + 1))  # +1 to include end
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
    parser = argparse.ArgumentParser(description='Run benchmark experiments with different hyperparameters')
    parser.add_argument('--num-hidden-layers', type=int, default=6,
                        help='Number of hidden layers (default: 6)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Size of each hidden layer (default: 128)')
    parser.add_argument('--seeds', nargs='+', default=['42'],
                        help='Seeds to run (can be individual seeds, comma-separated, or ranges like "63-100"). Default: 42')
    
    args = parser.parse_args()
    
    # Parse seeds (can be individual seeds, comma-separated, or ranges)
    run_seeds = parse_seeds(args.seeds)
    
    if not run_seeds:
        print("Error: No valid seeds specified.")
        exit(1)
    
    run_benchmark(num_hidden_layers=args.num_hidden_layers, hidden_dim=args.hidden_dim, run_seeds=run_seeds)

