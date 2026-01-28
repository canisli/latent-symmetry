#!/usr/bin/env python3
"""
Plot benchmark results from benchmark_penalties.py.

Plots loss as a function of lambda with separate lines for each penalized layer.
Supports aggregating across multiple seeds with 95% confidence interval error bars.

Usage:
    python scripts/plot_benchmark.py results/benchmark/
    python scripts/plot_benchmark.py results/benchmark/*.csv
    python scripts/plot_benchmark.py file1.csv file2.csv --layers 123l
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_csv_files(paths: list) -> pd.DataFrame:
    """
    Load CSV files from paths (can be files or directories).
    
    Args:
        paths: List of file paths or directory paths.
    
    Returns:
        Combined DataFrame.
    """
    csv_files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            csv_files.extend(path.glob("benchmark_*.csv"))
        elif path.exists() and path.suffix == '.csv':
            csv_files.append(path)
    
    if not csv_files:
        print(f"No CSV files found in {paths}")
        sys.exit(1)
    
    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
    
    if not dfs:
        print("No valid CSV files loaded.")
        sys.exit(1)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} experiments from {len(csv_files)} file(s)")
    return combined


def compute_stats(values: list) -> tuple:
    """Compute mean and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = np.mean(values)
    if n == 1:
        return mean, 0.0
    std = np.std(values, ddof=1)
    sem = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci = t_crit * sem
    return mean, ci


def aggregate_by_layer_lambda(df: pd.DataFrame) -> dict:
    """
    Aggregate data by (penalized_layer, lambda_sym).
    
    Returns:
        Dict mapping (layer, lambda) -> {'loss': [values], 'avg_Q': [values]}
    """
    agg = defaultdict(lambda: {'loss': [], 'avg_Q': []})
    
    # Compute avg_Q from per-layer columns if needed
    q_cols = [c for c in df.columns if c.startswith('Q_layer_') and not c.startswith('Q_h_')]
    
    for _, row in df.iterrows():
        layer = row.get('penalized_layer', '')
        if pd.isna(layer) or layer == '':
            layer = None
        else:
            layer = int(layer)
        
        lambda_val = row['lambda_sym']
        loss = row['best_val_loss']
        
        # Compute avg_Q
        if q_cols:
            q_vals = [row[c] for c in q_cols if pd.notna(row[c])]
            avg_q = np.mean(q_vals) if q_vals else np.nan
        else:
            avg_q = row.get('avg_Q', np.nan)
        
        agg[(layer, lambda_val)]['loss'].append(loss)
        agg[(layer, lambda_val)]['avg_Q'].append(avg_q)
    
    return agg


def parse_layer_spec(spec: str) -> set:
    """
    Parse layer specification string.
    
    Args:
        spec: String like "n123456l" where:
            - 'n' = None (baseline)
            - '1'-'9' = layer numbers
            - 'l' = layer -1 (output)
    
    Returns:
        Set of layer values to include.
    """
    layers = set()
    for c in spec.lower():
        if c == 'n':
            layers.add(None)
        elif c.isdigit():
            layers.add(int(c))
        elif c == 'l':
            layers.add(-1)
    return layers


def plot_loss_vs_lambda(df: pd.DataFrame, output_path: Path, layers_to_plot: set = None):
    """
    Plot loss and Q metric as functions of lambda for each penalized layer.
    """
    agg = aggregate_by_layer_lambda(df)
    
    # Compute stats per (layer, lambda)
    layer_data = defaultdict(lambda: {'lambda': [], 'loss_mean': [], 'loss_ci': [],
                                       'q_mean': [], 'q_ci': []})
    
    for (layer, lambda_val), values in agg.items():
        loss_mean, loss_ci = compute_stats(values['loss'])
        q_mean, q_ci = compute_stats(values['avg_Q'])
        
        layer_data[layer]['lambda'].append(lambda_val)
        layer_data[layer]['loss_mean'].append(loss_mean)
        layer_data[layer]['loss_ci'].append(loss_ci)
        layer_data[layer]['q_mean'].append(q_mean)
        layer_data[layer]['q_ci'].append(q_ci)
    
    # Filter layers if specified
    if layers_to_plot:
        layer_data = {k: v for k, v in layer_data.items() if k in layers_to_plot}
    
    if not layer_data:
        print("No data to plot after filtering.")
        return
    
    # Sort layers: numeric first, then -1, then None
    def layer_sort_key(x):
        if x is None:
            return (2, 0)
        elif x == -1:
            return (1, 0)
        else:
            return (0, x)
    
    sorted_layers = sorted(layer_data.keys(), key=layer_sort_key)
    num_layers = len(sorted_layers)
    
    print(f"Plotting {num_layers} layers: {sorted_layers}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    # Collect all lambda values for baseline horizontal line range
    all_lambdas = []
    for layer in sorted_layers:
        if layer is not None:
            all_lambdas.extend([l for l in layer_data[layer]['lambda'] if l > 0])
    
    for idx, layer in enumerate(sorted_layers):
        data = layer_data[layer]
        lambdas = np.array(data['lambda'])
        loss_mean = np.array(data['loss_mean'])
        loss_ci = np.array(data['loss_ci'])
        q_mean = np.array(data['q_mean'])
        q_ci = np.array(data['q_ci'])
        
        label = 'Baseline' if layer is None else f'Layer {layer}'
        
        if layer is None:
            # Baseline: horizontal line with shaded CI
            if len(loss_mean) > 0 and all_lambdas:
                x_min, x_max = min(all_lambdas), max(all_lambdas)
                ax1.axhline(y=loss_mean[0], color=colors[idx], linestyle='--',
                           linewidth=1.5, label=label)
                ax1.fill_between([x_min, x_max], loss_mean[0] - loss_ci[0],
                                loss_mean[0] + loss_ci[0], color=colors[idx], alpha=0.2)
                
                ax2.axhline(y=q_mean[0], color=colors[idx], linestyle='--',
                           linewidth=1.5, label=label)
                ax2.fill_between([x_min, x_max], q_mean[0] - q_ci[0],
                                q_mean[0] + q_ci[0], color=colors[idx], alpha=0.2)
        else:
            # Sort by lambda
            sort_idx = np.argsort(lambdas)
            lambdas = lambdas[sort_idx]
            loss_mean = loss_mean[sort_idx]
            loss_ci = loss_ci[sort_idx]
            q_mean = q_mean[sort_idx]
            q_ci = q_ci[sort_idx]
            
            # Filter to positive lambdas for log scale
            mask = lambdas > 0
            if np.any(mask):
                ax1.errorbar(lambdas[mask], loss_mean[mask], yerr=loss_ci[mask],
                            fmt='o-', linewidth=1.5, markersize=4, capsize=3,
                            label=label, color=colors[idx])
                ax2.errorbar(lambdas[mask], q_mean[mask], yerr=q_ci[mask],
                            fmt='o-', linewidth=1.5, markersize=4, capsize=3,
                            label=label, color=colors[idx])
    
    # Configure axes
    ax1.set_xlabel('λ (penalty weight)')
    ax1.set_ylabel('Best Validation Loss')
    ax1.set_title('Task Loss vs Penalty Strength')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('λ (penalty weight)')
    ax2.set_ylabel('Average Q')
    ax2.set_title('Symmetry Metric vs Penalty Strength')
    ax2.set_xscale('log')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Extract info for suptitle
    if 'num_hidden_layers' in df.columns and 'hidden_dim' in df.columns:
        n_layers = df['num_hidden_layers'].iloc[0]
        hidden_dim = df['hidden_dim'].iloc[0]
        n_seeds = df['seed'].nunique()
        fig.suptitle(f"Model: {n_layers}×{hidden_dim} MLP, {n_seeds} seed(s)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot benchmark results: loss vs lambda for each penalized layer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/plot_benchmark.py results/benchmark/
    python scripts/plot_benchmark.py results/benchmark/*.csv
    python scripts/plot_benchmark.py file1.csv file2.csv --layers n123l
    
Layer specification (--layers):
    n = baseline (no penalty)
    1-9 = layer numbers
    l = output layer (-1)
    
    Example: --layers n123l plots baseline + layers 1,2,3 + output
        """
    )
    
    parser.add_argument('paths', nargs='+', help='CSV files or directories containing benchmark results')
    parser.add_argument('--layers', type=str, default=None,
                        help='Layer spec: n=baseline, 1-9=layers, l=output (e.g., "n123l")')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: loss_vs_lambda.png in first input dir)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_csv_files(args.paths)
    
    # Parse layers
    layers_to_plot = parse_layer_spec(args.layers) if args.layers else None
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        first_path = Path(args.paths[0])
        if first_path.is_dir():
            output_path = first_path / 'loss_vs_lambda.png'
        else:
            output_path = first_path.parent / 'loss_vs_lambda.png'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plot_loss_vs_lambda(df, output_path, layers_to_plot)


if __name__ == '__main__':
    main()
