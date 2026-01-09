#!/usr/bin/env python3
"""
Test high lambda values to find the breaking point where task performance degrades.

Tests lambda = 100, 1000, 10000 on layer 1 to see if the model eventually fails to learn.
"""

import torch
import numpy as np
from pathlib import Path
from train import run_training, load_efp_preset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Load EFP preset
    config_dir = Path(__file__).parent / "config"
    edges_list = load_efp_preset('deg3', str(config_dir))
    
    # Test configurations
    lambda_values = [0.0, 1.0, 10.0, 100.0, 1000.0]
    
    print("="*80)
    print("HIGH LAMBDA TEST: Finding the Breaking Point")
    print("="*80)
    print(f"Lambda values to test: {lambda_values}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    results = []
    
    for lambda_val in lambda_values:
        symmetry_enabled = lambda_val > 0
        symmetry_layer = 1 if symmetry_enabled else None
        
        print(f"\n{'='*60}")
        print(f"Testing lambda = {lambda_val}")
        print(f"{'='*60}")
        
        result = run_training(
            # Data params - smaller for faster testing
            num_events=10000,
            n_particles=128,
            batch_size=256,
            input_scale=0.9515689,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            edges_list=edges_list,
            # Training params
            num_epochs=50,  # Reduced for testing
            learning_rate=1e-4,
            warmup_epochs=3,
            weight_decay=0.0,
            grad_clip=1.0,
            dropout=0.0,
            early_stopping_patience=10,
            # Model params - 4+4 architecture
            model_type='deepsets',
            hidden_channels=128,
            num_phi_layers=4,
            num_rho_layers=4,
            pool_mode='sum',
            # Symmetry params
            symmetry_enabled=symmetry_enabled,
            symmetry_layer=symmetry_layer,
            lambda_sym_max=lambda_val,
            std_eta=0.5,
            # Other
            run_seed=42,
            headless=True,
        )
        
        results.append({
            'lambda': lambda_val,
            'test_task_loss': result['test_task_loss'],
            'test_relative_rmse': result['test_relative_rmse'],
            'test_sym_loss': result.get('test_sym_loss'),
            'epochs_trained': result.get('epochs_trained'),
        })
        
        print(f"\nResult for lambda={lambda_val}:")
        print(f"  Test Task Loss:    {result['test_task_loss']:.4e}")
        print(f"  Test Rel RMSE:     {result['test_relative_rmse']:.4f}")
        if result.get('test_sym_loss') is not None:
            print(f"  Test Sym Loss:     {result['test_sym_loss']:.4e}")
        print(f"  Epochs trained:    {result.get('epochs_trained')}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: High Lambda Test Results")
    print("="*80)
    print(f"{'Lambda':<10} {'Task Loss':<15} {'Rel RMSE':<12} {'Sym Loss':<15} {'Epochs':<8}")
    print("-"*60)
    
    baseline_rmse = None
    for r in results:
        sym_str = f"{r['test_sym_loss']:.4e}" if r['test_sym_loss'] is not None else "N/A"
        print(f"{r['lambda']:<10} {r['test_task_loss']:<15.4e} {r['test_relative_rmse']:<12.4f} {sym_str:<15} {r['epochs_trained']:<8}")
        
        if r['lambda'] == 0.0:
            baseline_rmse = r['test_relative_rmse']
    
    print("-"*60)
    
    if baseline_rmse:
        print("\nDegradation vs Baseline:")
        for r in results:
            if r['lambda'] > 0:
                degradation = (r['test_relative_rmse'] - baseline_rmse) / baseline_rmse * 100
                print(f"  lambda={r['lambda']}: {degradation:+.1f}% {'(WORSE)' if degradation > 10 else '(OK)'}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("- If Rel RMSE stays similar across all lambda values → symmetry loss doesn't hurt")
    print("- If Rel RMSE degrades significantly at high lambda → found the breaking point")
    print("- Very low symmetry loss with good task loss → model learned invariant representations")
    print("="*80)


if __name__ == '__main__':
    main()

