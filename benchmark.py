import train
import csv
import sys

def run_experiment(symmetry_layer, lambda_sym_max, learning_rate, num_hidden_layers, csv_filename, fieldnames):
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
        run_seed=42
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

def run_benchmark(num_hidden_layers=6):
    """
    Run training experiments with different lambda_sym and symmetry_layer values.
    Collects results and outputs to both console and CSV file.
    """
    # Experiment configuration
    lambda_sym_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 100.0]
    symmetry_layers = list(range(1, num_hidden_layers + 1)) + [-1]  # Layers 1 to num_hidden_layers, plus -1
    learning_rate = 3e-4
    
    csv_filename = f'{num_hidden_layers}_hidden_layers_benchmark_results.csv'
    fieldnames = ['learning_rate', 'lambda_sym_max', 'symmetry_layer', 'test_task_loss', 'test_sym_loss']
    
    print("Starting benchmark experiments...")
    print(f"Lambda_sym values: {lambda_sym_values}")
    print(f"Symmetry layers: {symmetry_layers} + None")
    print(f"Learning rate: {learning_rate}")
    print(f"Total experiments: {len(lambda_sym_values) * len(symmetry_layers) + 1}")  # +1 for None case
    print(f"Results will be saved incrementally to {csv_filename}\n")
    
    # Open CSV file and write header
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Run experiments with symmetry layers
    for lambda_sym_max in lambda_sym_values:
        for symmetry_layer in symmetry_layers:
            run_experiment(symmetry_layer, lambda_sym_max, learning_rate, num_hidden_layers, csv_filename, fieldnames)
    
    # Run baseline experiment with symmetry_layer=None
    run_experiment(None, 0.0, learning_rate, num_hidden_layers, csv_filename, fieldnames)

    print(f"\nAll results have been saved to {csv_filename}")


if __name__ == '__main__':
    num_hidden_layers = 6
    if len(sys.argv) > 1:
        num_hidden_layers = int(sys.argv[1])
    run_benchmark(num_hidden_layers)

