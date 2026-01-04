#!/usr/bin/env python3
"""
Script to plot task rel RMSE and symmetry loss as a function of lambda_sym_max for all symmetry layers.
Can accept multiple CSV files to plot error bars across seeds.

Usage:
    python compare_lambdas.py <csv_file1> [csv_file2] [csv_file3] ...
    python compare_lambdas.py results/*.csv
    python compare_lambdas.py results/*.csv --layers 123456789l

Example:
    python compare_lambdas.py results/model=deepsets_layers=4+4_lr=3e-4_seed=42.csv
    python compare_lambdas.py results/model=deepsets_layers=4+4_lr=3e-4_seed=*.csv
"""

import csv
import sys
import re
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy import stats


def read_csv_file(csv_file: str) -> dict:
    """
    Read a CSV file and return layer data.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary mapping layer -> {'lambda': [...], 'task_loss': [...], 'sym_loss': [...], 'rel_rmse': [...]}
    """
    layer_data = defaultdict(lambda: {'lambda': [], 'task_loss': [], 'sym_loss': [], 'rel_rmse': []})
    
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
                task_loss = float(row['test_task_loss']) if row['test_task_loss'] else None
                rel_rmse = float(row['test_rel_rmse']) if row.get('test_rel_rmse') else None
                sym_loss_str = row['test_sym_loss']
                
                layer_data[row_layer]['lambda'].append(lambda_sym_max)
                layer_data[row_layer]['task_loss'].append(task_loss)
                layer_data[row_layer]['rel_rmse'].append(rel_rmse)
                
                # Handle symmetry loss (can be empty for None layer)
                if sym_loss_str == '' or sym_loss_str is None:
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


def aggregate_across_seeds(csv_files: list) -> dict:
    """
    Read multiple CSV files and aggregate data across seeds.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        Dictionary mapping (layer, lambda) -> {'task_loss': [...], 'sym_loss': [...], 'rel_rmse': [...]}
    """
    aggregated = defaultdict(lambda: {'task_loss': [], 'sym_loss': [], 'rel_rmse': []})
    
    for csv_file in csv_files:
        layer_data = read_csv_file(csv_file)
        for layer, data in layer_data.items():
            for lambda_val, task_loss, sym_loss, rel_rmse in zip(
                data['lambda'], data['task_loss'], data['sym_loss'], data['rel_rmse']
            ):
                key = (layer, lambda_val)
                if task_loss is not None:
                    aggregated[key]['task_loss'].append(task_loss)
                if sym_loss is not None:
                    aggregated[key]['sym_loss'].append(sym_loss)
                if rel_rmse is not None:
                    aggregated[key]['rel_rmse'].append(rel_rmse)
    
    return aggregated


def extract_model_info_from_filename(filename: str) -> dict:
    """
    Extract model info from CSV filename.
    
    Args:
        filename: CSV filename like "results/model=deepsets_layers=4+4_lr=3e-4_seed=42.csv"
    
    Returns:
        Dictionary with model, layers, lr, seed
    """
    basename = os.path.basename(filename)
    info = {}
    
    # Extract model type
    match = re.search(r'model=(\w+)', basename)
    if match:
        info['model'] = match.group(1)
    
    # Extract layers (e.g., "4+4" for DeepSets)
    match = re.search(r'layers=([\d+]+|\d+)', basename)
    if match:
        info['layers'] = match.group(1)
    
    # Extract learning rate
    match = re.search(r'lr=([\d.e+-]+)', basename)
    if match:
        info['lr'] = match.group(1)
    
    # Extract seed
    match = re.search(r'seed=(\d+)', basename)
    if match:
        info['seed'] = match.group(1)
    
    return info


def parse_layer_spec(layer_spec: str) -> set:
    """
    Parse layer specification string.
    
    Args:
        layer_spec: String like "n123456789l" where:
            - 'n' = None layer
            - '1'-'9' = layers 1-9
            - 'l' = layer -1
    
    Returns:
        Set of layer numbers/None to plot
    """
    layers_to_plot = set()
    
    for char in layer_spec.lower():
        if char == 'n':
            layers_to_plot.add(None)
        elif char in '123456789':
            layers_to_plot.add(int(char))
        elif char == 'l':
            layers_to_plot.add(-1)
        else:
            print(f"Warning: Unknown layer specifier '{char}'. Ignoring.")
    
    return layers_to_plot


def compute_stats(values: list) -> tuple:
    """
    Compute mean and 95% confidence interval.
    
    Args:
        values: List of numeric values
        
    Returns:
        Tuple of (mean, ci_width)
    """
    if not values:
        return None, None
    
    n = len(values)
    mean = np.mean(values)
    
    if n > 1:
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, n - 1)
        ci = t_critical * sem
    else:
        ci = 0.0
    
    return mean, ci


def plot_all_layers(csv_files: list, layers_to_plot: set = None):
    """
    Plot task rel RMSE and symmetry loss as a function of lambda_sym_max for specified layers.
    If multiple CSV files are provided, plots error bars across seeds.
    
    Args:
        csv_files: List of CSV file paths with benchmark results
        layers_to_plot: Set of layers to plot (None means plot all layers)
    """
    # Extract info from first filename for title
    info = extract_model_info_from_filename(csv_files[0])
    
    # Aggregate data across seeds
    aggregated = aggregate_across_seeds(csv_files)
    
    # Group by layer and compute statistics
    layer_stats = defaultdict(lambda: {
        'lambda': [], 'task_mean': [], 'task_ci': [],
        'sym_mean': [], 'sym_ci': [], 'rmse_mean': [], 'rmse_ci': []
    })
    
    for (layer, lambda_val), values in aggregated.items():
        task_mean, task_ci = compute_stats(values['task_loss'])
        sym_mean, sym_ci = compute_stats(values['sym_loss'])
        rmse_mean, rmse_ci = compute_stats(values['rel_rmse'])
        
        layer_stats[layer]['lambda'].append(lambda_val)
        layer_stats[layer]['task_mean'].append(task_mean)
        layer_stats[layer]['task_ci'].append(task_ci)
        layer_stats[layer]['sym_mean'].append(sym_mean)
        layer_stats[layer]['sym_ci'].append(sym_ci)
        layer_stats[layer]['rmse_mean'].append(rmse_mean)
        layer_stats[layer]['rmse_ci'].append(rmse_ci)
    
    if not layer_stats:
        print("Error: No data found in CSV files.")
        sys.exit(1)
    
    # Filter layers if specified
    if layers_to_plot is not None:
        layer_stats = {k: v for k, v in layer_stats.items() if k in layers_to_plot}
        if not layer_stats:
            print("Error: No matching layers found for specification.")
            sys.exit(1)
    
    # Sort layers: regular layers first (1-9), then -1, then None
    regular_layers = sorted([k for k in layer_stats.keys() if k not in (-1, None) and k is not None])
    sorted_layers = regular_layers + ([-1] if -1 in layer_stats else []) + ([None] if None in layer_stats else [])
    
    num_layers = len(sorted_layers)
    print(f"Found {num_layers} layers: {sorted_layers}")
    print(f"Processing {len(csv_files)} CSV file(s)")
    
    # Show data completeness
    print(f"\n{'='*60}")
    print("Data completeness per layer and lambda:")
    print(f"{'='*60}")
    
    for layer in sorted_layers:
        layer_str = "None" if layer is None else str(layer)
        lambda_values = sorted(set(layer_stats[layer]['lambda']))
        
        print(f"\nLayer {layer_str}:")
        for lambda_val in lambda_values:
            key = (layer, lambda_val)
            if key in aggregated:
                n_samples = len(aggregated[key]['rel_rmse'])
                status = "✓" if n_samples == len(csv_files) else "⚠"
                print(f"  {status} lambda={lambda_val:8.3f}: {n_samples}/{len(csv_files)} seeds")
    
    print(f"\n{'='*60}\n")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate colors using viridis colormap
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_layers))
    
    # Collect all lambda values for x-axis range (excluding None layer)
    all_lambdas = []
    for layer in sorted_layers:
        if layer is not None:
            all_lambdas.extend([l for l in layer_stats[layer]['lambda'] if l > 0])
    
    for idx, layer in enumerate(sorted_layers):
        lambda_values = np.array(layer_stats[layer]['lambda'])
        rmse_means = np.array(layer_stats[layer]['rmse_mean'])
        rmse_cis = np.array(layer_stats[layer]['rmse_ci'])
        sym_means = np.array(layer_stats[layer]['sym_mean'])
        sym_cis = np.array(layer_stats[layer]['sym_ci'])
        
        layer_str = "None" if layer is None else str(layer)
        
        if layer is None:
            # For Layer None, plot as horizontal line with shaded confidence interval
            if len(rmse_means) > 0 and rmse_means[0] is not None:
                rmse_mean = rmse_means[0]
                rmse_ci = rmse_cis[0] if rmse_cis[0] is not None else 0
                
                if all_lambdas:
                    x_min = min(all_lambdas)
                    x_max = max(all_lambdas)
                    
                    ax1.axhline(y=rmse_mean, color=colors[idx], linestyle='--',
                               linewidth=1.5, label=f'Layer {layer_str}', alpha=0.8)
                    ax1.fill_between([x_min, x_max],
                                     rmse_mean - rmse_ci,
                                     rmse_mean + rmse_ci,
                                     color=colors[idx], alpha=0.15)
        else:
            # Sort by lambda for proper line plot
            sort_idx = np.argsort(lambda_values)
            lambda_values = lambda_values[sort_idx]
            rmse_means = rmse_means[sort_idx]
            rmse_cis = rmse_cis[sort_idx]
            sym_means = sym_means[sort_idx]
            sym_cis = sym_cis[sort_idx]
            
            # Filter out zero/None lambda for log scale
            valid_rmse = (lambda_values > 0) & np.array([r is not None for r in rmse_means])
            valid_sym = (lambda_values > 0) & np.array([s is not None for s in sym_means])
            
            # Plot task rel RMSE
            if np.any(valid_rmse):
                rmse_means_valid = np.array([r if r is not None else np.nan for r in rmse_means])
                rmse_cis_valid = np.array([c if c is not None else 0 for c in rmse_cis])
                ax1.errorbar(
                    lambda_values[valid_rmse], rmse_means_valid[valid_rmse],
                    yerr=rmse_cis_valid[valid_rmse],
                    fmt='o-', linewidth=1.5, markersize=4, capsize=3,
                    label=f'Layer {layer_str}', color=colors[idx]
                )
            
            # Plot symmetry loss
            if np.any(valid_sym):
                sym_means_valid = np.array([s if s is not None else np.nan for s in sym_means])
                sym_cis_valid = np.array([c if c is not None else 0 for c in sym_cis])
                ax2.errorbar(
                    lambda_values[valid_sym], sym_means_valid[valid_sym],
                    yerr=sym_cis_valid[valid_sym],
                    fmt='s-', linewidth=1.5, markersize=4, capsize=3,
                    label=f'Layer {layer_str}', color=colors[idx]
                )
    
    # Configure task rel RMSE plot
    ax1.set_xlabel('λ_sym_max', fontsize=11)
    ax1.set_ylabel('Test Rel RMSE', fontsize=11)
    ax1.set_title('Task Rel RMSE vs Lambda', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8, loc='best')
    ax1.tick_params(labelsize=9)
    
    # Configure symmetry loss plot
    ax2.set_xlabel('λ_sym_max', fontsize=11)
    ax2.set_ylabel('Test Symmetry Loss', fontsize=11)
    ax2.set_title('Symmetry Loss vs Lambda', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8, loc='best')
    ax2.tick_params(labelsize=9)
    
    # Set suptitle
    suptitle_parts = []
    if info.get('model'):
        suptitle_parts.append(f"Model: {info['model']}")
    if info.get('layers'):
        suptitle_parts.append(f"Layers: {info['layers']}")
    if info.get('lr'):
        suptitle_parts.append(f"LR: {info['lr']}")
    suptitle_parts.append(f"Seeds: {len(csv_files)}")
    
    fig.suptitle(" | ".join(suptitle_parts), fontsize=12)
    
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_lambdas.py <csv_file1> [csv_file2] ... [--layers LAYER_SPEC]")
        print("\nLayer specification:")
        print("  Use a string like 'n123456789l' where:")
        print("    - 'n' = None layer (baseline)")
        print("    - '1'-'9' = layers 1-9")
        print("    - 'l' = layer -1 (output)")
        print("\nExamples:")
        print("  python compare_lambdas.py results/model=deepsets_layers=4+4_lr=3e-4_seed=42.csv")
        print("  python compare_lambdas.py results/*.csv --layers n5l")
        print("  python compare_lambdas.py results/*.csv --layers 123456789l")
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
            # Handle glob patterns
            expanded = glob.glob(sys.argv[i])
            if expanded:
                csv_files.extend(expanded)
            else:
                csv_files.append(sys.argv[i])
            i += 1
    
    if not csv_files:
        print("Error: No CSV files specified.")
        sys.exit(1)
    
    # Sort files for consistent ordering
    csv_files = sorted(set(csv_files))
    
    plot_all_layers(csv_files, layers_to_plot)


if __name__ == '__main__':
    main()

