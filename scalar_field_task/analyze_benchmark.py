import csv
from collections import defaultdict
import sys

def analyze_benchmark_results(num_hidden_layers=6):
    """
    Read benchmark results and find the best lambda_sym for each symmetry layer.
    Outputs the test task loss and test symmetry loss for the best configuration.
    """
    # Read CSV file
    csv_filename = f'{num_hidden_layers}_hidden_layers_benchmark_results.csv'
    results = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Handle None symmetry_layer (empty string in CSV)
            symmetry_layer = None if row['symmetry_layer'] == '' else int(row['symmetry_layer'])
            results.append({
                'learning_rate': float(row['learning_rate']),
                'lambda_sym_max': float(row['lambda_sym_max']),
                'symmetry_layer': symmetry_layer,
                'test_task_loss': float(row['test_task_loss']),
                'test_sym_loss': float(row['test_sym_loss']) if row['test_sym_loss'] else None
            })
    
    # Group by symmetry_layer and find best and runner-up lambda_sym_max for each
    best_results = {}
    runner_up_results = {}
    
    for result in results:
        layer = result['symmetry_layer']
        
        # Initialize or update best and runner-up results for this layer
        if layer not in best_results:
            best_results[layer] = result
        else:
            # Update best if this result is better
            if result['test_task_loss'] < best_results[layer]['test_task_loss']:
                # Current best becomes runner-up
                runner_up_results[layer] = best_results[layer]
                best_results[layer] = result
            elif layer not in runner_up_results or result['test_task_loss'] < runner_up_results[layer]['test_task_loss']:
                # This is better than current runner-up (or no runner-up exists)
                runner_up_results[layer] = result
    
    # Sort by symmetry_layer, but put None and -1 at the end (None last)
    regular_layers = sorted([k for k in best_results.keys() if k != -1 and k is not None])
    sorted_layers = regular_layers + ([-1] if -1 in best_results else []) + ([None] if None in best_results else [])
    
    # Print results
    print("="*80)
    print(csv_filename)
    print("="*80)
    print()
    
    headers = ['Symmetry Layer', 'Best λ_sym', 'Best Task Loss', 'Best Sym Loss', 
                'Runner-up λ_sym', 'Runner-up Task Loss', 'Runner-up Sym Loss']
    col_widths = [18, 12, 16, 16, 14, 18, 18]
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for layer in sorted_layers:
        best = best_results[layer]
        runner_up = runner_up_results.get(layer)
        
        layer_str = "None" if best['symmetry_layer'] is None else str(best['symmetry_layer'])
        best_lambda_str = "N/A" if best['symmetry_layer'] is None else f"{best['lambda_sym_max']:.1f}"
        
        if runner_up:
            runner_up_lambda_str = "N/A" if runner_up['symmetry_layer'] is None else f"{runner_up['lambda_sym_max']:.1f}"
            runner_up_task_loss = f"{runner_up['test_task_loss']:.4e}"
            runner_up_sym_loss = "N/A" if runner_up['symmetry_layer'] is None else (f"{runner_up['test_sym_loss']:.4e}" if runner_up['test_sym_loss'] is not None else "N/A")
        else:
            runner_up_lambda_str = "N/A"
            runner_up_task_loss = "N/A"
            runner_up_sym_loss = "N/A"
        
        # Handle symmetry loss for None layer
        best_sym_loss_str = "N/A" if best['symmetry_layer'] is None else (f"{best['test_sym_loss']:.4e}" if best['test_sym_loss'] is not None else "N/A")
        
        row = [
            layer_str,
            best_lambda_str,
            f"{best['test_task_loss']:.4e}",
            best_sym_loss_str,
            runner_up_lambda_str,
            runner_up_task_loss,
            runner_up_sym_loss
        ]
        print(" | ".join(val.ljust(w) for val, w in zip(row, col_widths)))
    
    print()


if __name__ == '__main__':
    num_hidden_layers = 6
    if len(sys.argv) > 1:
        num_hidden_layers = int(sys.argv[1])
    analyze_benchmark_results(num_hidden_layers)

