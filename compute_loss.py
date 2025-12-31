#!/usr/bin/env python3
"""
Script to compute and print 95% confidence intervals for results in results_2layer/.
Aggregates data across seeds and computes statistics.

Usage:
    python compute_loss.py [csv_file1] [csv_file2] ...
    
    If no files specified, automatically finds all CSV files in results_2layer/
"""

import csv
import sys
import os
import glob
import numpy as np
from scipy import stats

def read_csv_file(csv_file):
    """
    Read a CSV file and return data.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary with 'task_loss', 'sym_loss_1', 'sym_loss_2' lists
    """
    data = {'task_loss': [], 'sym_loss_1': [], 'sym_loss_2': []}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_loss = float(row['test_task_loss'])
                sym_loss_1 = float(row['test_sym_loss_1'])
                sym_loss_2 = float(row['test_sym_loss_2'])
                
                data['task_loss'].append(task_loss)
                data['sym_loss_1'].append(sym_loss_1)
                data['sym_loss_2'].append(sym_loss_2)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing column '{e.args[0]}' in CSV file '{csv_file}'.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data format in CSV file '{csv_file}': {e}")
        sys.exit(1)
    
    return data

def aggregate_across_seeds(csv_files):
    """
    Read multiple CSV files and aggregate data across seeds.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        Dictionary with aggregated 'task_loss', 'sym_loss_1', 'sym_loss_2' lists
    """
    aggregated = {'task_loss': [], 'sym_loss_1': [], 'sym_loss_2': []}
    
    for csv_file in csv_files:
        data = read_csv_file(csv_file)
        # Each CSV file should have one row (since lambda values are fixed)
        if len(data['task_loss']) > 0:
            aggregated['task_loss'].append(data['task_loss'][0])
            aggregated['sym_loss_1'].append(data['sym_loss_1'][0])
            aggregated['sym_loss_2'].append(data['sym_loss_2'][0])
    
    return aggregated

def compute_95_ci(values):
    """
    Compute mean and 95% confidence interval for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Tuple of (mean, ci_lower, ci_upper, ci_width)
    """
    if len(values) == 0:
        return None, None, None, None
    
    n = len(values)
    mean = np.mean(values)
    
    if n > 1:
        std = np.std(values, ddof=1)  # Sample standard deviation
        sem = std / np.sqrt(n)  # Standard error of the mean
        t_critical = stats.t.ppf(0.975, n - 1)  # 95% CI, two-tailed
        ci_width = t_critical * sem
        ci_lower = mean - ci_width
        ci_upper = mean + ci_width
    else:
        ci_width = 0.0
        ci_lower = mean
        ci_upper = mean
    
    return mean, ci_lower, ci_upper, ci_width

def print_results(aggregated):
    """
    Print results with 95% confidence intervals.
    
    Args:
        aggregated: Dictionary with 'task_loss', 'sym_loss_1', 'sym_loss_2' lists
    """
    n_seeds = len(aggregated['task_loss'])
    
    print(f"{'='*70}")
    print(f"Results aggregated across {n_seeds} seed(s)")
    print(f"{'='*70}\n")
    
    # Compute statistics for each metric
    task_mean, task_ci_lower, task_ci_upper, task_ci_width = compute_95_ci(aggregated['task_loss'])
    sym1_mean, sym1_ci_lower, sym1_ci_upper, sym1_ci_width = compute_95_ci(aggregated['sym_loss_1'])
    sym2_mean, sym2_ci_lower, sym2_ci_upper, sym2_ci_width = compute_95_ci(aggregated['sym_loss_2'])
    
    # Print task loss
    print("Test Task Loss:")
    print(f"  Mean:     {task_mean:.6e}")
    if n_seeds > 1:
        print(f"  95% CI:   [{task_ci_lower:.6e}, {task_ci_upper:.6e}]")
        print(f"  CI Width: {task_ci_width:.6e}")
    else:
        print(f"  (Single sample - no CI)")
    print()
    
    # Print symmetry loss 1
    print("Test Symmetry Loss 1 (last hidden layer):")
    print(f"  Mean:     {sym1_mean:.6e}")
    if n_seeds > 1:
        print(f"  95% CI:   [{sym1_ci_lower:.6e}, {sym1_ci_upper:.6e}]")
        print(f"  CI Width: {sym1_ci_width:.6e}")
    else:
        print(f"  (Single sample - no CI)")
    print()
    
    # Print symmetry loss 2
    print("Test Symmetry Loss 2 (output layer):")
    print(f"  Mean:     {sym2_mean:.6e}")
    if n_seeds > 1:
        print(f"  95% CI:   [{sym2_ci_lower:.6e}, {sym2_ci_upper:.6e}]")
        print(f"  CI Width: {sym2_ci_width:.6e}")
    else:
        print(f"  (Single sample - no CI)")
    print()
    
    print(f"{'='*70}")

def main():
    # Get CSV files from command line or find all in results_2layer/
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    else:
        # Automatically find all CSV files in results_2layer/
        csv_files = glob.glob('results_2layer/*.csv')
        csv_files.sort()  # Sort for consistent ordering
    
    if not csv_files:
        print("Error: No CSV files found.")
        print("Usage: python compute_loss.py [csv_file1] [csv_file2] ...")
        print("       Or run without arguments to process all files in results_2layer/")
        sys.exit(1)
    
    print(f"Processing {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print()
    
    # Aggregate data across seeds
    aggregated = aggregate_across_seeds(csv_files)
    
    # Print results
    print_results(aggregated)

if __name__ == '__main__':
    main()

