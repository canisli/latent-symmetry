import train
import csv

def run_benchmark():
    """
    Run training experiments with different lambda_sym and symmetry_layer values.
    Collects results and outputs to both console and CSV file.
    """
    # Experiment configuration
    lambda_sym_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    symmetry_layers = [1, 2, 3, 4, -1]
    learning_rate = 3e-4
    
    results = []
    
    print("Starting benchmark experiments...")
    print(f"Lambda_sym values: {lambda_sym_values}")
    print(f"Symmetry layers: {symmetry_layers}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total experiments: {len(lambda_sym_values) * len(symmetry_layers)}\n")
    
    # Run experiments
    for lambda_sym_max in lambda_sym_values:
        for symmetry_layer in symmetry_layers:
            print(f"Running: lambda_sym_max={lambda_sym_max}, symmetry_layer={symmetry_layer}")
            
            # For lambda_sym=0.0, we can set symmetry_layer to None
            # But let's keep it consistent and use the layer anyway
            result = train.main(
                headless=True,
                symmetry_layer=symmetry_layer,
                lambda_sym_max=lambda_sym_max,
                lambda_sym_min=0.0,
                learning_rate=learning_rate,
                seed=42
            )
            
            results.append({
                'learning_rate': result['learning_rate'],
                'lambda_sym_max': result['lambda_sym_max'],
                'symmetry_layer': result['symmetry_layer'],
                'test_task_loss': result['test_task_loss'],
                'test_sym_loss': result['test_sym_loss'] if result['test_sym_loss'] is not None else 0.0
            })
    
    # Sort results by lambda_sym_max, then symmetry_layer
    results.sort(key=lambda x: (x['lambda_sym_max'], x['symmetry_layer']))
    
    # Print formatted table
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    
    # Format table manually
    headers = ['Learning Rate', 'Lambda_sym_max', 'Symmetry Layer', 'Test Task Loss', 'Test Symmetry Loss']
    col_widths = [15, 15, 15, 18, 20]
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for r in results:
        row = [
            f"{r['learning_rate']:.2e}",
            f"{r['lambda_sym_max']:.1f}",
            str(r['symmetry_layer']),
            f"{r['test_task_loss']:.4e}",
            f"{r['test_sym_loss']:.4e}" if r['test_sym_loss'] is not None else "N/A"
        ]
        print(" | ".join(val.ljust(w) for val, w in zip(row, col_widths)))
    
    # Save to CSV
    csv_filename = 'benchmark_results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['learning_rate', 'lambda_sym_max', 'symmetry_layer', 'test_task_loss', 'test_sym_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in results:
            writer.writerow({
                'learning_rate': r['learning_rate'],
                'lambda_sym_max': r['lambda_sym_max'],
                'symmetry_layer': r['symmetry_layer'],
                'test_task_loss': r['test_task_loss'],
                'test_sym_loss': r['test_sym_loss'] if r['test_sym_loss'] is not None else ''
            })
    
    print(f"\nResults saved to {csv_filename}")


if __name__ == '__main__':
    run_benchmark()

