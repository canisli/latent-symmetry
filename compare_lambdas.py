#!/usr/bin/env python3
"""
Script to plot loss as a function of lambda_sym_max for all symmetry layers.
Can accept multiple CSV files to plot error bars across seeds.

Usage:
    python compare_lambdas.py <csv_file1> [csv_file2] [csv_file3] ...

Example:
    python compare_lambdas.py layers=6_seed=42.csv
    python compare_lambdas.py layers=6_seed=42.csv layers=6_seed=43.csv layers=6_seed=44.csv
"""

import csv
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy import stats

def read_csv_file(csv_file):
    """
    Read a CSV file and return layer data.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary mapping layer -> {'lambda': [...], 'task_loss': [...], 'sym_loss': [...]}
    """
    layer_data = defaultdict(lambda: {'lambda': [], 'task_loss': [], 'sym_loss': []})
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse symmetry_layer (can be empty string for None)
                row_layer = row['symmetry_layer']
                if row_layer == '':
                    row_layer = None
                else:
                    row_layer = int(row_layer)
                
                lambda_sym_max = float(row['lambda_sym_max'])
                task_loss = float(row['test_task_loss'])
                sym_loss_str = row['test_sym_loss']
                
                layer_data[row_layer]['lambda'].append(lambda_sym_max)
                layer_data[row_layer]['task_loss'].append(task_loss)
                
                # Handle symmetry loss (can be empty for None layer)
                if sym_loss_str == '':
                    layer_data[row_layer]['sym_loss'].append(None)
                else:
                    layer_data[row_layer]['sym_loss'].append(float(sym_loss_str))
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing column '{e.args[0]}' in CSV file '{csv_file}'.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data format in CSV file '{csv_file}': {e}")
        sys.exit(1)
    
    return layer_data

def aggregate_across_seeds(csv_files):
    """
    Read multiple CSV files and aggregate data across seeds.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        Dictionary mapping (layer, lambda) -> {'task_loss': [...], 'sym_loss': [...]}
    """
    # Aggregate data: (layer, lambda) -> list of values across seeds
    aggregated = defaultdict(lambda: {'task_loss': [], 'sym_loss': []})
    
    for csv_file in csv_files:
        layer_data = read_csv_file(csv_file)
        for layer, data in layer_data.items():
            for lambda_val, task_loss, sym_loss in zip(data['lambda'], data['task_loss'], data['sym_loss']):
                key = (layer, lambda_val)
                aggregated[key]['task_loss'].append(task_loss)
                if sym_loss is not None:
                    aggregated[key]['sym_loss'].append(sym_loss)
    
    return aggregated

def extract_layers_from_filename(filename):
    """
    Extract layers specification from CSV filename.
    
    Args:
        filename: CSV filename like "layers=6x128_lr=3e-4_seed=42.csv" or "results/layers=6x128_lr=3e-4_seed=42.csv"
    
    Returns:
        String like "6x128" or None if not found
    """
    # Extract the basename if it's a path
    basename = filename.split('/')[-1]
    
    # Match pattern: layers=<num>x<dim>_ or layers=<num>_
    match = re.search(r'layers=(\d+x\d+|\d+)', basename)
    if match:
        return match.group(1)
    return None

def extract_learning_rate_from_filename(filename):
    """
    Extract learning rate from CSV filename.
    
    Args:
        filename: CSV filename like "layers=6x128_lr=3e-4_seed=42.csv" or "results/layers=6x128_lr=3e-4_seed=42.csv"
    
    Returns:
        String like "3e-4" or None if not found
    """
    # Extract the basename if it's a path
    basename = filename.split('/')[-1]
    
    # Match pattern: lr=<value>_ or lr=<value>.csv
    match = re.search(r'lr=([\d.e-]+)', basename)
    if match:
        return match.group(1)
    return None

def parse_layer_spec(layer_spec):
    """
    Parse layer specification string.
    
    Args:
        layer_spec: String like "n123456l" where:
            - 'n' = None layer
            - '1'-'6' = layers 1-6
            - 'l' = layer -1
    
    Returns:
        Set of layer numbers/None to plot
    """
    layers_to_plot = set()
    
    for char in layer_spec.lower():
        if char == 'n':
            layers_to_plot.add(None)
        elif char in '123456':
            layers_to_plot.add(int(char))
        elif char == 'l':
            layers_to_plot.add(-1)
        else:
            print(f"Warning: Unknown layer specifier '{char}'. Ignoring.")
    
    return layers_to_plot

def plot_all_layers(csv_files, layers_to_plot=None):
    """
    Plot task loss and symmetry loss as a function of lambda_sym_max for specified layers.
    If multiple CSV files are provided, plots error bars across seeds.
    
    Args:
        csv_files: List of CSV file paths with benchmark results
        layers_to_plot: Set of layers to plot (None means plot all layers)
    """
    # Extract layers and learning rate from first filename for suptitle
    layers_title = extract_layers_from_filename(csv_files[0])
    lr_title = extract_learning_rate_from_filename(csv_files[0])
    
    # Aggregate data across seeds
    aggregated = aggregate_across_seeds(csv_files)
    
    # Group by layer and compute statistics
    layer_stats = defaultdict(lambda: {'lambda': [], 'task_mean': [], 'task_ci': [], 
                                       'sym_mean': [], 'sym_ci': []})
    
    for (layer, lambda_val), values in aggregated.items():
        task_losses = values['task_loss']
        sym_losses = values['sym_loss']
        
        n = len(task_losses)
        task_mean = np.mean(task_losses)
        # Calculate 95% confidence interval for mean
        if n > 1:
            task_std = np.std(task_losses, ddof=1)  # Sample standard deviation
            sem = task_std / np.sqrt(n)  # Standard error of the mean
            t_critical = stats.t.ppf(0.975, n - 1)  # 95% CI, two-tailed
            task_ci = t_critical * sem
        else:
            task_ci = 0.0  # No confidence interval with only 1 sample
        
        layer_stats[layer]['lambda'].append(lambda_val)
        layer_stats[layer]['task_mean'].append(task_mean)
        layer_stats[layer]['task_ci'].append(task_ci)
        
        if sym_losses:
            sym_mean = np.mean(sym_losses)
            if len(sym_losses) > 1:
                sym_std = np.std(sym_losses, ddof=1)
                sym_sem = sym_std / np.sqrt(len(sym_losses))
                sym_t_critical = stats.t.ppf(0.975, len(sym_losses) - 1)
                sym_ci = sym_t_critical * sym_sem
            else:
                sym_ci = 0.0  # No confidence interval with only 1 sample
            layer_stats[layer]['sym_mean'].append(sym_mean)
            layer_stats[layer]['sym_ci'].append(sym_ci)
        else:
            layer_stats[layer]['sym_mean'].append(None)
            layer_stats[layer]['sym_ci'].append(None)
    
    if not layer_stats:
        print(f"Error: No data found in CSV files.")
        sys.exit(1)
    
    # Filter layers if specified
    if layers_to_plot is not None:
        layer_stats = {k: v for k, v in layer_stats.items() if k in layers_to_plot}
        if not layer_stats:
            print(f"Error: No matching layers found for specification.")
            sys.exit(1)
    
    # Sort layers: regular layers first, then -1, then None
    regular_layers = sorted([k for k in layer_stats.keys() if k != -1 and k is not None])
    sorted_layers = regular_layers + ([-1] if -1 in layer_stats else []) + ([None] if None in layer_stats else [])
    
    num_layers = len(sorted_layers)
    print(f"Found {num_layers} layers: {sorted_layers}")
    print(f"Processing {len(csv_files)} CSV file(s)")
    
    # Debug: Show how many seeds contributed data for each lambda value per layer
    print(f"\n{'='*60}")
    print("Data completeness per layer and lambda:")
    print(f"{'='*60}")
    
    all_lambda_values = set()
    for layer in sorted_layers:
        all_lambda_values.update(layer_stats[layer]['lambda'])
    
    # Count how many seeds contributed to each (layer, lambda) combination
    for layer in sorted_layers:
        layer_str = "None" if layer is None else str(layer)
        lambda_values = sorted(set(layer_stats[layer]['lambda']))
        
        print(f"\nLayer {layer_str}:")
        for lambda_val in lambda_values:
            key = (layer, lambda_val)
            if key in aggregated:
                n_samples = len(aggregated[key]['task_loss'])
                status = "✓" if n_samples == len(csv_files) else "⚠"
                print(f"  {status} lambda={lambda_val:6.1f}: {n_samples}/{len(csv_files)} seeds")
    
    print(f"\n{'='*60}\n")
    
    # Create two plots: one for task loss, one for symmetry loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate gradient colors for each layer (using viridis colormap for smooth gradient)
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    # Collect all lambda values for x-axis range
    all_lambdas = []
    for layer in sorted_layers:
        if layer is not None:
            all_lambdas.extend(layer_stats[layer]['lambda'])
    
    for idx, layer in enumerate(sorted_layers):
        lambda_values = np.array(layer_stats[layer]['lambda'])
        task_means = np.array(layer_stats[layer]['task_mean'])
        task_cis = np.array(layer_stats[layer]['task_ci'])
        sym_means = layer_stats[layer]['sym_mean']
        sym_cis = layer_stats[layer]['sym_ci']
        
        layer_str = "None" if layer is None else str(layer)
        
        if layer is None:
            # For Layer None, plot as horizontal line with shaded confidence interval band
            if len(task_means) > 0:
                task_mean = task_means[0]  # Should be the same for all lambda values
                task_ci = task_cis[0]
                
                if all_lambdas:
                    x_min = min(all_lambdas)
                    x_max = max(all_lambdas)
                    
                    # Plot horizontal line
                    ax1.axhline(y=task_mean, color=colors[idx], linestyle='--', 
                               linewidth=1.5, label=f'Layer {layer_str}')
                    
                    # Shade 95% confidence interval area
                    ax1.fill_between([x_min, x_max], 
                                     task_mean - task_ci, 
                                     task_mean + task_ci,
                                     color=colors[idx], alpha=0.2)
        else:
            # Sort by lambda_sym_max for plotting
            sort_idx = np.argsort(lambda_values)
            lambda_values = lambda_values[sort_idx]
            task_means = task_means[sort_idx]
            task_cis = task_cis[sort_idx]
            sym_means = [sym_means[i] for i in sort_idx]
            sym_cis = [sym_cis[i] for i in sort_idx]
            
            # Plot task loss with 95% confidence interval error bars
            ax1.errorbar(lambda_values, task_means, yerr=task_cis, 
                        fmt='o-', linewidth=1.5, markersize=4, capsize=3,
                        label=f'Layer {layer_str}', color=colors[idx])
            
            # Plot symmetry loss with 95% confidence interval error bars
            if any(s is not None for s in sym_means):
                sym_means_array = np.array([s if s is not None else np.nan for s in sym_means])
                sym_cis_array = np.array([s if s is not None else np.nan for s in sym_cis])
                
                # Filter out NaN values for plotting
                valid_mask = ~np.isnan(sym_means_array)
                if np.any(valid_mask):
                    ax2.errorbar(lambda_values[valid_mask], sym_means_array[valid_mask], 
                               yerr=sym_cis_array[valid_mask],
                               fmt='s-', linewidth=1.5, markersize=4, capsize=3,
                               label=f'Layer {layer_str}', color=colors[idx])
    
    # Configure task loss plot
    ax1.set_xlabel('λ_sym_max', fontsize=10)
    ax1.set_ylabel('Test Task Loss', fontsize=10)
    ax1.set_title('Task Loss', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=8)
    
    # Configure symmetry loss plot
    ax2.set_xlabel('λ_sym_max', fontsize=10)
    ax2.set_ylabel('Test Symmetry Loss', fontsize=10)
    ax2.set_title('Symmetry Loss', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)
    
    # Set suptitle with layers and learning rate
    suptitle_parts = []
    if layers_title:
        suptitle_parts.append(f"Hidden Layers={layers_title}")
    if lr_title:
        suptitle_parts.append(f"Learning Rate={lr_title}")
    
    if suptitle_parts:
        fig.suptitle(", ".join(suptitle_parts), fontsize=12)

    
    # Save plot
    # output_file = csv_file.replace('.csv', '_lambda_comparison.png')
    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # print(f"\nPlot saved to {output_file}")
    
    # Also show plot
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_lambdas.py <csv_file1> [csv_file2] ... [--layers LAYER_SPEC]")
        print("\nLayer specification:")
        print("  Use a string like 'n123456l' where:")
        print("    - 'n' = None layer")
        print("    - '1'-'6' = layers 1-6")
        print("    - 'l' = layer -1")
        print("\nExample:")
        print("  python compare_lambdas.py layers=6_seed=42.csv")
        print("  python compare_lambdas.py layers=6_seed=42.csv layers=6_seed=43.csv --layers n123")
        print("  python compare_lambdas.py layers=6_seed=42.csv --layers 456l")
        sys.exit(1)
    
    # Parse arguments
    csv_files = []
    layers_to_plot = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--layers' and i + 1 < len(sys.argv):
            layers_to_plot = parse_layer_spec(sys.argv[i + 1])
            i += 2
        else:
            csv_files.append(sys.argv[i])
            i += 1
    
    if not csv_files:
        print("Error: No CSV files specified.")
        sys.exit(1)
    
    plot_all_layers(csv_files, layers_to_plot)

if __name__ == '__main__':
    main()

