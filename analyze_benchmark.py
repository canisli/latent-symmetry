import csv
from collections import defaultdict

def analyze_benchmark_results(csv_filename='benchmark_results.csv'):
    """
    Read benchmark results and find the best lambda_sym for each symmetry layer.
    Outputs the test task loss and test symmetry loss for the best configuration.
    """
    # Read CSV file
    results = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append({
                'learning_rate': float(row['learning_rate']),
                'lambda_sym_max': float(row['lambda_sym_max']),
                'symmetry_layer': int(row['symmetry_layer']),
                'test_task_loss': float(row['test_task_loss']),
                'test_sym_loss': float(row['test_sym_loss']) if row['test_sym_loss'] else None
            })
    
    # Group by symmetry_layer and find best lambda_sym_max for each
    best_results = {}
    
    for result in results:
        layer = result['symmetry_layer']
        
        # Initialize or update best result for this layer
        if layer not in best_results:
            best_results[layer] = result
        else:
            # Keep the one with lower test_task_loss
            if result['test_task_loss'] < best_results[layer]['test_task_loss']:
                best_results[layer] = result
    
    # Sort by symmetry_layer, but put -1 at the end
    sorted_layers = sorted([k for k in best_results.keys() if k != -1]) + ([-1] if -1 in best_results else [])
    
    # Print results
    print("="*80)
    print("BEST LAMBDA_SYM FOR EACH SYMMETRY LAYER")
    print("="*80)
    print()
    
    headers = ['Symmetry Layer', 'Best Lambda_sym_max', 'Test Task Loss', 'Test Symmetry Loss']
    col_widths = [18, 20, 18, 20]
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for layer in sorted_layers:
        result = best_results[layer]
        row = [
            str(result['symmetry_layer']),
            f"{result['lambda_sym_max']:.1f}",
            f"{result['test_task_loss']:.4e}",
            f"{result['test_sym_loss']:.4e}" if result['test_sym_loss'] is not None else "N/A"
        ]
        print(" | ".join(val.ljust(w) for val, w in zip(row, col_widths)))
    
    print()


if __name__ == '__main__':
    analyze_benchmark_results()

