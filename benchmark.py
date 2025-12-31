import train
import csv
import argparse
import os

def run_experiment(lambda_sym_1_max, lambda_sym_2_max, learning_rate, num_hidden_layers, hidden_dim, run_seed, csv_filename, fieldnames):
    """
    Run a single training experiment and save results.
    
    Returns:
        result_dict: Dictionary with experiment results
    """
    print(f"Running: lambda_sym_1_max={lambda_sym_1_max}, lambda_sym_2_max={lambda_sym_2_max}")
    
    result = train.main(
        headless=True,
        lambda_sym_1_max=lambda_sym_1_max,
        lambda_sym_2_max=lambda_sym_2_max,
        learning_rate=learning_rate,
        num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        run_seed=run_seed
    )
    
    result_dict = {
        'learning_rate': result['learning_rate'],
        'lambda_sym_1_max': result['lambda_sym_1_max'],
        'lambda_sym_2_max': result['lambda_sym_2_max'],
        'test_task_loss': result['test_task_loss'],
        'test_sym_loss_1': result['test_sym_loss_1'],
        'test_sym_loss_2': result['test_sym_loss_2']
    }
    
    # Write result immediately to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'learning_rate': result_dict['learning_rate'],
            'lambda_sym_1_max': result_dict['lambda_sym_1_max'],
            'lambda_sym_2_max': result_dict['lambda_sym_2_max'],
            'test_task_loss': result_dict['test_task_loss'],
            'test_sym_loss_1': result_dict['test_sym_loss_1'],
            'test_sym_loss_2': result_dict['test_sym_loss_2']
        })
    
    # Print results
    print(f"\033[32m  -> Task loss: {result_dict['test_task_loss']:.4e}, Sym loss 1: {result_dict['test_sym_loss_1']:.4e}, Sym loss 2: {result_dict['test_sym_loss_2']:.4e}\033[0m")
    print(f"\033[32m  -> Saved to {csv_filename}\033[0m\n")
    
    return result_dict

def run_benchmark(num_hidden_layers=6, hidden_dim=128, run_seeds=None):
    """
    Run training experiments with fixed lambda_sym_1_max=1.0, lambda_sym_2_max=1.0, and learning_rate=1e-4.
    Can run with multiple seeds, saving each to a separate CSV file.
    
    Args:
        num_hidden_layers: Number of hidden layers
        hidden_dim: Size of each hidden layer
        run_seeds: List of seeds to run (each seed gets its own CSV file). Defaults to range(42, 101) if None.
    """
    # Default to many seeds if not specified
    if run_seeds is None:
        run_seeds = list(range(42, 101))
    
    # Fixed experiment configuration
    lambda_sym_1_max = 0.5
    lambda_sym_2_max = 0.5
    learning_rate = 1e-4
    
    # Create results_2layer directory if it doesn't exist
    os.makedirs('results_2layer', exist_ok=True)
    
    print("Starting benchmark experiments...")
    print(f"Lambda_sym_1_max: {lambda_sym_1_max}")
    print(f"Lambda_sym_2_max: {lambda_sym_2_max}")
    print(f"Learning rate: {learning_rate}")
    print(f"Seeds: {run_seeds}")
    print(f"Total experiments: {len(run_seeds)}\n")
    
    # Run for each seed
    for seed_idx, run_seed in enumerate(run_seeds):
        lr_string = f'{learning_rate:.0e}'.replace('-0','-')
        csv_filename = f'results_2layer/layers={num_hidden_layers}x{hidden_dim}_lr={lr_string}_seed={run_seed}.csv'
        fieldnames = ['learning_rate', 'lambda_sym_1_max', 'lambda_sym_2_max', 'test_task_loss', 'test_sym_loss_1', 'test_sym_loss_2']
        
        # Check if row with these lambda values already exists
        row_exists = False
        if os.path.exists(csv_filename):
            with open(csv_filename, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if lambda values match (compare as floats)
                    if (float(row['lambda_sym_1_max']) == lambda_sym_1_max and 
                        float(row['lambda_sym_2_max']) == lambda_sym_2_max):
                        row_exists = True
                        break
        
        if row_exists:
            print(f"\n{'='*60}")
            print(f"Skipping seed {run_seed} (row with lambda_sym_1_max={lambda_sym_1_max}, lambda_sym_2_max={lambda_sym_2_max} already exists in {csv_filename})")
            print(f"{'='*60}\n")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running seed {run_seed} ({seed_idx + 1}/{len(run_seeds)})")
        print(f"{'='*60}")
        print(f"Results will be saved incrementally to {csv_filename}\n")
        
        # Create CSV file with header if it doesn't exist
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        
        # Run experiment with fixed lambda values
        run_experiment(lambda_sym_1_max, lambda_sym_2_max, learning_rate, num_hidden_layers, hidden_dim, run_seed, csv_filename, fieldnames)

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
    parser.add_argument('--seeds', nargs='+', default=None,
                        help='Seeds to run (can be individual seeds, comma-separated, or ranges like "63-100"). Default: 42-100')
    
    args = parser.parse_args()
    
    # Parse seeds if provided, otherwise use default range
    if args.seeds is not None:
        run_seeds = parse_seeds(args.seeds)
        if not run_seeds:
            print("Error: No valid seeds specified.")
            exit(1)
    else:
        run_seeds = None  # Will default to range(42, 101) in run_benchmark
    
    run_benchmark(num_hidden_layers=args.num_hidden_layers, hidden_dim=args.hidden_dim, run_seeds=run_seeds)

