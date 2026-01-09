#!/usr/bin/env python3
"""
Analyze per-KP relative RMSE across different symmetry configurations.

This script:
1. Loads results from CSV files
2. Compares per-KP performance across baseline and symmetry-trained models
3. Identifies which KPs are most affected by symmetry training
"""

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load_results(csv_pattern: str = "results/model=deepsets_layers=3+3_lr=1e-4_kps=deg3_seed=*.csv"):
    """Load results from multiple CSV files."""
    files = glob.glob(csv_pattern)
    if not files:
        print(f"No files found matching pattern: {csv_pattern}")
        return None
    
    print(f"Loading {len(files)} result files...")
    
    # Aggregate data: (lambda, layer) -> {metric: [values across seeds]}
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for csv_file in files:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lambda_val = float(row['lambda_sym_max'])
                    layer_str = row['symmetry_layer']
                    layer = None if layer_str == '' else int(layer_str)
                    
                    key = (lambda_val, layer)
                    
                    # Store per-KP RMSEs
                    for kp_idx in range(1, 6):
                        col = f'test_per_kp_rel_rmse_kp{kp_idx}'
                        if col in row and row[col]:
                            aggregated[key][f'kp{kp_idx}'].append(float(row[col]))
                    
                    # Store overall metrics
                    if row.get('test_rel_rmse'):
                        aggregated[key]['overall'].append(float(row['test_rel_rmse']))
                    if row.get('test_sym_loss'):
                        aggregated[key]['sym_loss'].append(float(row['test_sym_loss']))
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return aggregated


def analyze_per_kp(aggregated):
    """Analyze per-KP performance across configurations."""
    
    # Edge configurations for deg3
    edges_list = [
        "[(0,1), (0,1), (0,1)]",  # degree 3, 2-particle
        "[(0,1), (0,1), (1,2)]",  # degree 3, 3-particle chain
        "[(0,1), (1,2), (0,2)]",  # degree 3, 3-particle triangle
        "[(0,1), (1,2), (2,3)]",  # degree 3, 4-particle chain
        "[(0,1), (0,2), (0,3)]",  # degree 3, 4-particle star
    ]
    
    print("\n" + "="*80)
    print("PER-KP PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Get baseline performance (lambda=0, layer=None)
    baseline_key = (0.0, None)
    if baseline_key not in aggregated:
        print("Warning: No baseline data found")
        return
    
    baseline = aggregated[baseline_key]
    print(f"\nBaseline (no symmetry) - {len(baseline.get('overall', []))} seeds:")
    print("-"*60)
    for kp_idx in range(1, 6):
        kp_key = f'kp{kp_idx}'
        if kp_key in baseline:
            mean = np.mean(baseline[kp_key])
            std = np.std(baseline[kp_key])
            print(f"  KP{kp_idx}: {mean:.4f} ± {std:.4f}  {edges_list[kp_idx-1]}")
    print(f"  Overall: {np.mean(baseline['overall']):.4f} ± {np.std(baseline['overall']):.4f}")
    
    # Analyze layer 1 symmetry at different lambda values
    print("\n" + "="*80)
    print("LAYER 1 SYMMETRY COMPARISON")
    print("="*80)
    
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    # Collect data for plotting
    plot_data = {f'KP{i}': {'baseline': [], 'lambda': [], 'mean': [], 'std': []} for i in range(1, 6)}
    plot_data['Overall'] = {'baseline': [], 'lambda': [], 'mean': [], 'std': []}
    
    for lambda_val in lambda_values:
        key = (lambda_val, 1)  # Layer 1
        if key not in aggregated:
            continue
            
        data = aggregated[key]
        n_seeds = len(data.get('overall', []))
        
        print(f"\nLambda = {lambda_val} ({n_seeds} seeds):")
        print("-"*60)
        
        for kp_idx in range(1, 6):
            kp_key = f'kp{kp_idx}'
            if kp_key in data and kp_key in baseline:
                mean = np.mean(data[kp_key])
                std = np.std(data[kp_key])
                baseline_mean = np.mean(baseline[kp_key])
                change = (mean - baseline_mean) / baseline_mean * 100
                
                print(f"  KP{kp_idx}: {mean:.4f} ± {std:.4f}  ({change:+.1f}% vs baseline)")
                
                plot_data[f'KP{kp_idx}']['lambda'].append(lambda_val)
                plot_data[f'KP{kp_idx}']['mean'].append(mean)
                plot_data[f'KP{kp_idx}']['std'].append(std)
                plot_data[f'KP{kp_idx}']['baseline'].append(baseline_mean)
        
        if 'overall' in data:
            mean = np.mean(data['overall'])
            std = np.std(data['overall'])
            baseline_mean = np.mean(baseline['overall'])
            change = (mean - baseline_mean) / baseline_mean * 100
            print(f"  Overall: {mean:.4f} ± {std:.4f}  ({change:+.1f}% vs baseline)")
            
            plot_data['Overall']['lambda'].append(lambda_val)
            plot_data['Overall']['mean'].append(mean)
            plot_data['Overall']['std'].append(std)
            plot_data['Overall']['baseline'].append(baseline_mean)
    
    # Analyze which KPs are most affected
    print("\n" + "="*80)
    print("KP SENSITIVITY ANALYSIS")
    print("="*80)
    print("\nWhich KPs are most affected by layer 1 symmetry training?")
    print("-"*60)
    
    # For lambda=10 (strong symmetry)
    key = (10.0, 1)
    if key in aggregated:
        data = aggregated[key]
        changes = []
        for kp_idx in range(1, 6):
            kp_key = f'kp{kp_idx}'
            if kp_key in data and kp_key in baseline:
                mean = np.mean(data[kp_key])
                baseline_mean = np.mean(baseline[kp_key])
                change = (mean - baseline_mean) / baseline_mean * 100
                changes.append((kp_idx, change, edges_list[kp_idx-1]))
        
        # Sort by change magnitude
        changes.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nAt lambda=10.0, sorted by impact:")
        for kp_idx, change, edges in changes:
            impact = "HIGH" if abs(change) > 10 else "LOW"
            print(f"  KP{kp_idx}: {change:+.1f}% [{impact}]  {edges}")
    
    # Create visualization
    create_per_kp_plot(plot_data, edges_list)
    
    return plot_data


def create_per_kp_plot(plot_data, edges_list):
    """Create visualization of per-KP performance."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, 5))
    
    for idx, kp_name in enumerate(['KP1', 'KP2', 'KP3', 'KP4', 'KP5', 'Overall']):
        ax = axes[idx]
        data = plot_data[kp_name]
        
        if len(data['lambda']) > 0:
            lambda_vals = np.array(data['lambda'])
            means = np.array(data['mean'])
            stds = np.array(data['std'])
            baseline = data['baseline'][0] if data['baseline'] else 0
            
            # Plot with error bars
            ax.errorbar(lambda_vals, means, yerr=stds, marker='o', linewidth=2, 
                       capsize=5, label='Layer 1 sym')
            
            # Plot baseline
            ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2, 
                      label=f'Baseline ({baseline:.4f})')
            
            ax.set_xscale('log')
            ax.set_xlabel('Lambda')
            ax.set_ylabel('Relative RMSE')
            
            if kp_name.startswith('KP'):
                kp_idx = int(kp_name[2]) - 1
                ax.set_title(f'{kp_name}\n{edges_list[kp_idx]}')
            else:
                ax.set_title(kp_name)
            
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('per_kp_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved per-KP analysis plot to per_kp_analysis.png")
    plt.show()


def main():
    # Load results
    aggregated = load_results("results/model=deepsets_layers=3+3_lr=1e-4_kps=deg3_seed=*.csv")
    
    if aggregated is None:
        print("No results to analyze.")
        return
    
    # Analyze per-KP performance
    analyze_per_kp(aggregated)
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Key insights:
1. KPs with MORE particle correlations should be harder to learn with constrained layer 1
2. Simple 2-particle KPs (KP1) might be easier than complex multi-particle ones (KP4, KP5)
3. Triangle graphs (KP3) involve all pairwise interactions - may be most affected
4. Star graphs (KP5) have one particle connecting to all others - tests center-of-mass features
5. If all KPs degrade equally, the symmetry constraint affects all equally
""")


if __name__ == '__main__':
    main()

