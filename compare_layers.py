#!/usr/bin/env python3
"""
Script to plot test task loss as a function of symmetry layer.
Filters for lambda_sym_max=1.0 and mu_head=1.0.
Can accept multiple CSV files to plot error bars across seeds.

Usage:
    python compare_layers.py <csv_file1> [csv_file2] [csv_file3] ...

Example:
    python compare_layers.py results/layers=6x128_lr=3e-4_seed=42.csv
    python compare_layers.py results/layers=6x128_lr=3e-4_seed=42.csv results/layers=6x128_lr=3e-4_seed=43.csv
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
    Read a CSV file and return layer data filtered for lambda_sym_max=1.0 and mu_head=1.0.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary mapping layer -> {'task_loss': [...], 'sym_loss': [...]}
    """
    layer_data = defaultdict(lambda: {'task_loss': [], 'sym_loss': []})
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for lambda_sym_max=1.0 and mu_head=1.0
                lambda_sym_max = float(row['lambda_sym_max'])
                mu_head = float(row.get('mu_head', '0.0'))  # Handle old CSV files without mu_head
                
                if lambda_sym_max != 1.0 or mu_head != 1.0:
                    continue
                
                # Parse symmetry_layer (can be empty string for None)
                row_layer = row['symmetry_layer']
                if row_layer == '':
                    row_layer = None
                else:
                    row_layer = int(row_layer)
                
                task_loss = float(row['test_task_loss'])
                sym_loss_str = row['test_sym_loss']
                
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
        Dictionary mapping layer -> {'task_loss': [...], 'sym_loss': [...]}
    """
    # Aggregate data: layer -> list of values across seeds
    aggregated = defaultdict(lambda: {'task_loss': [], 'sym_loss': []})
    
    for csv_file in csv_files:
        layer_data = read_csv_file(csv_file)
        for layer, data in layer_data.items():
            for task_loss, sym_loss in zip(data['task_loss'], data['sym_loss']):
                aggregated[layer]['task_loss'].append(task_loss)
                if sym_loss is not None:
                    aggregated[layer]['sym_loss'].append(sym_loss)
    
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
    Plot test task loss as a function of symmetry layer.
    Filters for lambda_sym_max=1.0 and mu_head=1.0.
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
    
    if not aggregated:
        print(f"Error: No data found matching lambda_sym_max=1.0 and mu_head=1.0 in CSV files.")
        sys.exit(1)
    
    # Compute statistics for each layer
    layer_stats = {}
    
    for layer, values in aggregated.items():
        task_losses = values['task_loss']
        
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
        
        layer_stats[layer] = {
            'task_mean': task_mean,
            'task_ci': task_ci,
            'n_samples': n
        }
    
    # Filter layers if specified
    if layers_to_plot is not None:
        layer_stats = {k: v for k, v in layer_stats.items() if k in layers_to_plot}
        if not layer_stats:
            print(f"Error: No matching layers found for specification.")
            sys.exit(1)
    
    # Extract None layer stats separately
    none_layer_stats = layer_stats.get(None)
    
    # Remove None layer from stats for x-axis plotting
    layer_stats_no_none = {k: v for k, v in layer_stats.items() if k is not None}
    
    # Sort layers: regular layers first, then -1 (no None layer)
    regular_layers = sorted([k for k in layer_stats_no_none.keys() if k != -1])
    sorted_layers = regular_layers + ([-1] if -1 in layer_stats_no_none else [])
    
    num_layers = len(sorted_layers)
    print(f"Found {num_layers} layers: {sorted_layers}")
    if none_layer_stats:
        print(f"Found None layer (baseline)")
    print(f"Processing {len(csv_files)} CSV file(s)")
    
    # Debug: Show how many seeds contributed data for each layer
    print(f"\n{'='*60}")
    print("Data completeness per layer:")
    print(f"{'='*60}")
    
    for layer in sorted_layers:
        layer_str = str(layer)
        n_samples = layer_stats_no_none[layer]['n_samples']
        status = "✓" if n_samples == len(csv_files) else "⚠"
        print(f"  {status} Layer {layer_str}: {n_samples}/{len(csv_files)} seeds")
    
    if none_layer_stats:
        n_samples = none_layer_stats['n_samples']
        status = "✓" if n_samples == len(csv_files) else "⚠"
        print(f"  {status} Layer None: {n_samples}/{len(csv_files)} seeds")
    
    print(f"\n{'='*60}\n")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Prepare data for plotting (excluding None layer)
    layer_labels = []
    layer_positions = []
    task_means = []
    task_cis = []
    
    for idx, layer in enumerate(sorted_layers):
        layer_str = str(layer)
        layer_labels.append(layer_str)
        layer_positions.append(idx)
        task_means.append(layer_stats_no_none[layer]['task_mean'])
        task_cis.append(layer_stats_no_none[layer]['task_ci'])
    
    # Convert to numpy arrays
    layer_positions = np.array(layer_positions)
    task_means = np.array(task_means)
    task_cis = np.array(task_cis)
    
    # Plot None layer as horizontal line with shaded confidence interval band (if available)
    if none_layer_stats is not None:
        none_layer_mean = none_layer_stats['task_mean']
        none_layer_ci = none_layer_stats['task_ci']
        
        if len(layer_positions) > 0:
            x_min = layer_positions[0] - 0.5
            x_max = layer_positions[-1] + 0.5
        else:
            x_min = -0.5
            x_max = 0.5
        
        # Plot horizontal line
        ax.axhline(y=none_layer_mean, color='orange', linestyle='--', 
                   linewidth=2, label='None (baseline)')
        
        # Shade 95% confidence interval area
        ax.fill_between([x_min, x_max], 
                        none_layer_mean - none_layer_ci, 
                        none_layer_mean + none_layer_ci,
                        color='orange', alpha=0.2)
    
    # Plot all layers with error bars
    ax.errorbar(layer_positions, 
                task_means, 
                yerr=task_cis,
                fmt='o-', linewidth=1.5, markersize=6, capsize=4,
                label='Layers', color='steelblue')
    
    # Configure plot
    ax.set_xlabel('Symmetry Layer', fontsize=12)
    ax.set_ylabel('Test Task Loss (95% CI)', fontsize=12)
    ax.set_title('Test Task Loss by Layer', fontsize=13)
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(layer_labels)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    
    # Set suptitle with layers and learning rate
    suptitle_parts = []
    if layers_title:
        suptitle_parts.append(f"Hidden Layers={layers_title}")
    if lr_title:
        suptitle_parts.append(f"Learning Rate={lr_title}")
    suptitle_parts.append("λ=1.0, μ=1.0")
    
    if suptitle_parts:
        fig.suptitle(", ".join(suptitle_parts), fontsize=12)
    
    # Show plot
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_layers.py <csv_file1> [csv_file2] ... [--layers LAYER_SPEC]")
        print("\nLayer specification:")
        print("  Use a string like 'n123456l' where:")
        print("    - 'n' = None layer")
        print("    - '1'-'6' = layers 1-6")
        print("    - 'l' = layer -1")
        print("\nExample:")
        print("  python compare_layers.py results/layers=6x128_lr=3e-4_seed=42.csv")
        print("  python compare_layers.py results/layers=6x128_lr=3e-4_seed=42.csv results/layers=6x128_lr=3e-4_seed=43.csv --layers n123")
        print("  python compare_layers.py results/layers=6x128_lr=3e-4_seed=42.csv --layers 456l")
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

