#!/usr/bin/env python3
"""
Plot benchmark results from benchmark_penalties.py.

Generates:
1. loss_vs_lambda.png - Loss as function of λ with lines for each penalized layer
2. Q_vs_layer/ folder - Q as function of layer for each (λ, penalized_layer) combo

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_csv_files(paths: list) -> pd.DataFrame:
    """Load CSV files from paths (can be files or directories)."""
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
    values = [v for v in values if not np.isnan(v)]
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


def parse_layer_spec(spec: str) -> set:
    """Parse layer spec: n=baseline, 1-9=layers, l=output (-1)."""
    layers = set()
    for c in spec.lower():
        if c == 'n':
            layers.add(None)
        elif c.isdigit():
            layers.add(int(c))
        elif c == 'l':
            layers.add(-1)
    return layers


def layer_sort_key(x):
    """Sort key: numeric layers first, then -1, then None."""
    if x is None:
        return (2, 0)
    elif x == -1:
        return (1, 0)
    else:
        return (0, x)


def plot_loss_vs_lambda(df: pd.DataFrame, output_path: Path, layers_to_plot: set = None):
    """Plot loss as function of lambda for each penalized layer."""
    # Aggregate by (penalized_layer, lambda)
    agg = defaultdict(list)
    for _, row in df.iterrows():
        layer = row.get('penalized_layer', '')
        layer = None if (pd.isna(layer) or layer == '') else int(layer)
        agg[(layer, row['lambda_sym'])].append(row['best_val_loss'])
    
    # Compute stats
    layer_data = defaultdict(lambda: {'lambda': [], 'mean': [], 'ci': []})
    for (layer, lambda_val), losses in agg.items():
        mean, ci = compute_stats(losses)
        layer_data[layer]['lambda'].append(lambda_val)
        layer_data[layer]['mean'].append(mean)
        layer_data[layer]['ci'].append(ci)
    
    # Filter layers
    if layers_to_plot:
        layer_data = {k: v for k, v in layer_data.items() if k in layers_to_plot}
    
    if not layer_data:
        print("No data to plot.")
        return
    
    sorted_layers = sorted(layer_data.keys(), key=layer_sort_key)
    print(f"Plotting {len(sorted_layers)} layers: {sorted_layers}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_layers)))
    
    # Get lambda range for baseline
    all_lambdas = []
    for layer in sorted_layers:
        if layer is not None:
            all_lambdas.extend([l for l in layer_data[layer]['lambda'] if l > 0])
    
    for idx, layer in enumerate(sorted_layers):
        data = layer_data[layer]
        lambdas = np.array(data['lambda'])
        means = np.array(data['mean'])
        cis = np.array(data['ci'])
        label = 'Baseline' if layer is None else f'Layer {layer}'
        
        if layer is None:
            # Baseline: horizontal line
            if len(means) > 0 and all_lambdas:
                x_min, x_max = min(all_lambdas), max(all_lambdas)
                ax.axhline(y=means[0], color=colors[idx], linestyle='--', linewidth=1.5, label=label)
                ax.fill_between([x_min, x_max], means[0] - cis[0], means[0] + cis[0],
                               color=colors[idx], alpha=0.2)
        else:
            # Sort by lambda
            sort_idx = np.argsort(lambdas)
            lambdas, means, cis = lambdas[sort_idx], means[sort_idx], cis[sort_idx]
            mask = lambdas > 0
            if np.any(mask):
                ax.errorbar(lambdas[mask], means[mask], yerr=cis[mask],
                           fmt='o-', linewidth=1.5, markersize=4, capsize=3,
                           label=label, color=colors[idx])
    
    ax.set_xlabel('λ (penalty weight)')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Task Loss vs Penalty Strength')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    if 'num_hidden_layers' in df.columns:
        n_layers = df['num_hidden_layers'].iloc[0]
        hidden_dim = df['hidden_dim'].iloc[0]
        n_seeds = df['seed'].nunique()
        fig.suptitle(f"Model: {n_layers}×{hidden_dim} MLP, {n_seeds} seed(s)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_Q_vs_layer_single(q_means: list, q_cis: list, layer_labels: list, 
                           save_path: Path, title: str, penalized_layer_idx: int = None,
                           oracle_q: float = None):
    """
    Plot Q vs layer using bar charts (linear and log scale).
    
    Based on latsym.metrics.plotting.plot_metric_vs_layer style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(layer_labels))
    color = 'steelblue'
    
    # Add oracle if provided
    if oracle_q is not None:
        layer_labels = layer_labels + ['oracle']
        q_means = q_means + [oracle_q]
        q_cis = q_cis + [0.0]
        colors = [color] * (len(layer_labels) - 1) + ['green']
        x = np.arange(len(layer_labels))
    else:
        colors = [color] * len(layer_labels)
    
    # Highlight penalized layer in red
    if penalized_layer_idx is not None and 0 <= penalized_layer_idx < len(colors):
        colors[penalized_layer_idx] = 'red'
    
    for ax_idx, (ax, scale) in enumerate([(axes[0], 'linear'), (axes[1], 'log')]):
        # Bar plot with error bars
        bars = ax.bar(x, q_means, color=colors, edgecolor='black', alpha=0.8)
        ax.errorbar(x, q_means, yerr=q_cis, fmt='none', ecolor='black', capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=45, ha='right')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Q (orbit variance ratio)')
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(scale.capitalize())
        
        if scale == 'linear':
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale('log')
            # Set floor for log scale
            min_val = min([v for v in q_means if v > 0], default=1e-6)
            ax.set_ylim(bottom=min_val * 0.5)
    
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_Q_vs_layer(df: pd.DataFrame, output_dir: Path):
    """
    For each (lambda, penalized_layer) combo, plot Q as function of layer index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Q column names, sorted by layer index
    q_cols = sorted([c for c in df.columns if c.startswith('Q_layer_') and not c.startswith('Q_h_')],
                    key=lambda x: (0, int(x.split('_')[-1])) if x != 'Q_layer_-1' else (1, 0))
    
    if not q_cols:
        print("No Q_layer columns found.")
        return
    
    # Extract layer indices and labels
    layer_indices = []
    layer_labels = []
    for c in q_cols:
        idx = int(c.replace('Q_layer_', ''))
        layer_indices.append(idx)
        layer_labels.append('output' if idx == -1 else f'layer_{idx}')
    
    # Get oracle Q if available
    oracle_q = df['oracle_Q'].dropna().mean() if 'oracle_Q' in df.columns else None
    
    # Group by (lambda_sym, penalized_layer)
    groups = df.groupby(['lambda_sym', 'penalized_layer'], dropna=False)
    
    n_plots = 0
    for (lambda_val, pen_layer), group in groups:
        if lambda_val == 0:
            continue  # Skip baseline
        
        pen_layer_int = None if pd.isna(pen_layer) else int(pen_layer)
        pen_layer_str = 'none' if pen_layer_int is None else pen_layer_int
        
        # Find index of penalized layer
        pen_idx = None
        if pen_layer_int is not None:
            try:
                pen_idx = layer_indices.index(pen_layer_int)
            except ValueError:
                pass
        
        # Aggregate Q values across seeds
        q_means = []
        q_cis = []
        for col in q_cols:
            values = group[col].dropna().tolist()
            mean, ci = compute_stats(values)
            q_means.append(mean)
            q_cis.append(ci)
        
        title = f'Q vs Layer  |  λ={lambda_val}, penalized={pen_layer_str}'
        filename = f'Q_lambda={lambda_val}_pen={pen_layer_str}.png'
        
        plot_Q_vs_layer_single(q_means, q_cis, layer_labels.copy(), output_dir / filename,
                               title, penalized_layer_idx=pen_idx, oracle_q=oracle_q)
        n_plots += 1
    
    # Plot baseline
    baseline = df[df['lambda_sym'] == 0]
    if len(baseline) > 0:
        q_means = []
        q_cis = []
        for col in q_cols:
            values = baseline[col].dropna().tolist()
            mean, ci = compute_stats(values)
            q_means.append(mean)
            q_cis.append(ci)
        
        plot_Q_vs_layer_single(q_means, q_cis, layer_labels.copy(), output_dir / 'Q_baseline.png',
                               'Q vs Layer  |  Baseline (no penalty)', oracle_q=oracle_q)
        n_plots += 1
    
    print(f"Saved {n_plots} Q vs layer plots to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/plot_benchmark.py results/benchmark/
    python scripts/plot_benchmark.py results/benchmark/*.csv --layers n123l
        """
    )
    
    parser.add_argument('paths', nargs='+', help='CSV files or directories')
    parser.add_argument('--layers', type=str, default=None,
                        help='Layer spec for loss plot: n=baseline, 1-9=layers, l=output')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: first input path)')
    
    args = parser.parse_args()
    
    df = load_csv_files(args.paths)
    layers_to_plot = parse_layer_spec(args.layers) if args.layers else None
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        first_path = Path(args.paths[0])
        output_dir = first_path if first_path.is_dir() else first_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_loss_vs_lambda(df, output_dir / 'loss_vs_lambda.png', layers_to_plot)
    plot_Q_vs_layer(df, output_dir / 'Q_vs_layer')


if __name__ == '__main__':
    main()
