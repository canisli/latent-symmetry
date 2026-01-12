#!/usr/bin/env python3
"""
Investigate why networks trained on non-Lorentz invariant EFPs still show
decreasing symmetry loss with depth.

Key question: If EFPs are NOT Lorentz invariant, why would the network
learn invariant intermediate representations?

Possible explanations:
1. The network is not predicting EFPs correctly
2. The output layer is NOT invariant (as expected), but intermediate layers are
3. Something about the EFP structure or data distribution

This script diagnoses the issue.
"""

import torch
import numpy as np
from pathlib import Path

from models import DeepSets
from data.kp_dataset import make_kp_dataloader, compute_kps
from symmetry import rand_lorentz
from train import load_efp_preset
import energyflow as ef

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path: str, num_kps: int = 5) -> DeepSets:
    """Load a trained DeepSets model."""
    model = DeepSets(
        in_channels=4,
        out_channels=num_kps,
        hidden_channels=128,
        num_phi_layers=4,
        num_rho_layers=4,
        pool_mode='sum',
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def investigate_prediction_invariance(model_path: str, n_events: int = 100, n_particles: int = 32):
    """
    Check if model predictions change under Lorentz transforms.
    
    For a model trained on non-invariant EFPs:
    - f(x) should approximate EFP(x)
    - f(Λx) should approximate EFP(Λx)  
    - Since EFP(x) ≠ EFP(Λx), we should have f(x) ≠ f(Λx)
    """
    print("=" * 70)
    print("INVESTIGATION: Model Prediction Invariance")
    print("=" * 70)
    
    edges_list = load_efp_preset('deg3', 'config')
    num_kps = len(edges_list)
    model = load_model(model_path, num_kps)
    
    # Generate test data
    np.random.seed(42)
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    X_scaled = X / 0.9515689  # Apply input scaling
    
    # Compute true EFP values (unnormed)
    Y_efp = compute_kps(X, edges_list, measure='eeefm', beta=2.0, kappa=1.0, normed=False)
    Y_efp_log = np.log1p(Y_efp)
    
    # Get model predictions on original data
    X_torch = torch.from_numpy(X_scaled).to(device)
    with torch.no_grad():
        Y_pred = model(X_torch).cpu().numpy()
    
    # Apply Lorentz transform
    torch.manual_seed(123)
    L = rand_lorentz(
        shape=torch.Size([n_events]),
        std_eta=0.5,
        device=device,
        dtype=torch.float32,
    )
    
    # Transform the scaled inputs
    X_transformed = torch.matmul(L.unsqueeze(1), X_torch.unsqueeze(-1)).squeeze(-1)
    
    # Get predictions on transformed data
    with torch.no_grad():
        Y_pred_transformed = model(X_transformed).cpu().numpy()
    
    # Also compute true EFPs on transformed data (need to unscale first)
    X_transformed_unscaled = X_transformed.cpu().numpy() * 0.9515689
    Y_efp_transformed = compute_kps(X_transformed_unscaled, edges_list, 
                                     measure='eeefm', beta=2.0, kappa=1.0, normed=False)
    Y_efp_transformed_log = np.log1p(Y_efp_transformed)
    
    print("\n1. TRUE EFP INVARIANCE CHECK")
    print("-" * 50)
    efp_diff = np.abs(Y_efp_log - Y_efp_transformed_log)
    efp_rel_diff = efp_diff / (np.abs(Y_efp_log) + 1e-10)
    print(f"   Mean absolute diff:  {np.mean(efp_diff):.4e}")
    print(f"   Mean relative diff:  {np.mean(efp_rel_diff):.4f}")
    print(f"   Max relative diff:   {np.max(efp_rel_diff):.4f}")
    if np.mean(efp_rel_diff) > 0.01:
        print("   → TRUE EFPs are NOT Lorentz invariant (as expected)")
    else:
        print("   → WARNING: EFPs appear invariant?!")
    
    print("\n2. MODEL PREDICTION QUALITY")
    print("-" * 50)
    pred_error = np.abs(Y_pred - Y_efp_log)
    pred_rel_error = pred_error / (np.abs(Y_efp_log) + 1e-10)
    print(f"   Mean absolute error:  {np.mean(pred_error):.4e}")
    print(f"   Mean relative error:  {np.mean(pred_rel_error):.4f}")
    if np.mean(pred_rel_error) < 0.1:
        print("   → Model is predicting EFPs reasonably well")
    else:
        print("   → WARNING: Model is NOT predicting EFPs well!")
    
    print("\n3. MODEL PREDICTION INVARIANCE")
    print("-" * 50)
    pred_diff = np.abs(Y_pred - Y_pred_transformed)
    pred_rel_diff = pred_diff / (np.abs(Y_pred) + np.abs(Y_pred_transformed) + 1e-10)
    print(f"   Mean absolute diff:  {np.mean(pred_diff):.4e}")
    print(f"   Mean relative diff:  {np.mean(pred_rel_diff):.4f}")
    print(f"   Max relative diff:   {np.max(pred_rel_diff):.4f}")
    if np.mean(pred_rel_diff) < 0.01:
        print("   → Model predictions ARE invariant (network learned invariance!)")
    else:
        print("   → Model predictions are NOT invariant (expected for EFP targets)")
    
    print("\n4. DOES THE MODEL TRACK EFP CHANGES?")
    print("-" * 50)
    # If model is good, Y_pred_transformed should approximate Y_efp_transformed_log
    track_error = np.abs(Y_pred_transformed - Y_efp_transformed_log)
    track_rel_error = track_error / (np.abs(Y_efp_transformed_log) + 1e-10)
    print(f"   Mean tracking error:  {np.mean(track_rel_error):.4f}")
    
    # Compare: does the change in predictions match the change in true EFPs?
    pred_change = Y_pred_transformed - Y_pred
    efp_change = Y_efp_transformed_log - Y_efp_log
    change_correlation = np.corrcoef(pred_change.flatten(), efp_change.flatten())[0, 1]
    print(f"   Correlation of changes: {change_correlation:.4f}")
    if change_correlation > 0.5:
        print("   → Model tracks EFP changes under transforms")
    else:
        print("   → Model does NOT track EFP changes well")
    
    return {
        'efp_rel_diff': np.mean(efp_rel_diff),
        'pred_rel_error': np.mean(pred_rel_error),
        'pred_rel_diff': np.mean(pred_rel_diff),
        'change_correlation': change_correlation,
    }


def investigate_layer_outputs(model_path: str, n_events: int = 50, n_particles: int = 32):
    """
    Look at what each layer outputs for original vs transformed inputs.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: Layer-wise Output Changes")
    print("=" * 70)
    
    edges_list = load_efp_preset('deg3', 'config')
    num_kps = len(edges_list)
    model = load_model(model_path, num_kps)
    
    # Generate test data
    np.random.seed(42)
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    X_scaled = X / 0.9515689
    X_torch = torch.from_numpy(X_scaled).to(device)
    
    # Apply Lorentz transform
    torch.manual_seed(456)
    L = rand_lorentz(
        shape=torch.Size([n_events]),
        std_eta=0.5,
        device=device,
        dtype=torch.float32,
    )
    X_transformed = torch.matmul(L.unsqueeze(1), X_torch.unsqueeze(-1)).squeeze(-1)
    
    print("\nRelative difference ||h(x) - h(Λx)||² / (||h(x)||² + ||h(Λx)||²) at each layer:")
    print("-" * 70)
    print(f"{'Layer':<15} {'Rel Diff':<15} {'Interpretation':<40}")
    print("-" * 70)
    
    # Check each layer
    layer_indices = list(range(1, 10)) + [-1]  # phi1-4, pool, rho1-4, output
    layer_names = ['phi_1', 'phi_2', 'phi_3', 'phi_4', 'pool', 
                   'rho_1', 'rho_2', 'rho_3', 'rho_4', 'output']
    
    with torch.no_grad():
        for idx, name in zip(layer_indices, layer_names):
            h1 = model.forward_with_intermediate(X_torch, idx)
            h2 = model.forward_with_intermediate(X_transformed, idx)
            
            # Compute relative difference
            diff_sq = (h1 - h2).pow(2).sum()
            norm_sq = h1.pow(2).sum() + h2.pow(2).sum() + 1e-8
            rel_diff = (diff_sq / norm_sq).item()
            
            if rel_diff < 0.01:
                interpretation = "INVARIANT"
            elif rel_diff < 0.1:
                interpretation = "Mostly invariant"
            elif rel_diff < 0.5:
                interpretation = "Partially variant"
            else:
                interpretation = "NOT invariant"
            
            print(f"{name:<15} {rel_diff:<15.6f} {interpretation:<40}")
    
    print("-" * 70)


def investigate_training_convergence(model_path: str):
    """
    Check if the model actually converged to predicting EFPs.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: Training Convergence Check")
    print("=" * 70)
    
    edges_list = load_efp_preset('deg3', 'config')
    num_kps = len(edges_list)
    model = load_model(model_path, num_kps)
    
    # Generate validation data
    np.random.seed(999)
    n_events = 500
    X = ef.gen_random_events_mcom(n_events, 128, dim=4).astype(np.float32)
    X_scaled = X / 0.9515689
    
    # Compute true EFP values
    Y_efp = compute_kps(X, edges_list, measure='eeefm', beta=2.0, kappa=1.0, normed=False)
    Y_efp_log = np.log1p(Y_efp)
    
    # Get model predictions
    X_torch = torch.from_numpy(X_scaled).to(device)
    with torch.no_grad():
        Y_pred = model(X_torch).cpu().numpy()
    
    print("\nPer-EFP prediction quality (relative RMSE):")
    print("-" * 50)
    for i in range(num_kps):
        y_true = Y_efp_log[:, i]
        y_pred = Y_pred[:, i]
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        std = np.std(y_true)
        rel_rmse = rmse / std if std > 0 else float('inf')
        print(f"  EFP {i+1}: rel_RMSE = {rel_rmse:.4f}")
    
    overall_rmse = np.sqrt(np.mean((Y_efp_log - Y_pred) ** 2))
    overall_std = np.std(Y_efp_log)
    overall_rel_rmse = overall_rmse / overall_std
    print(f"\nOverall: rel_RMSE = {overall_rel_rmse:.4f}")
    
    if overall_rel_rmse > 0.5:
        print("\n⚠️  WARNING: Model has NOT learned to predict EFPs well!")
        print("   This explains why symmetry loss decreases - the model")
        print("   is learning some OTHER function that happens to be invariant.")
    else:
        print("\n✓ Model has learned to predict EFPs reasonably")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='4x4_none_efps.pt')
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        return
    
    print(f"\nAnalyzing model: {args.model}")
    
    # Run all investigations
    investigate_training_convergence(args.model)
    results = investigate_prediction_invariance(args.model)
    investigate_layer_outputs(args.model)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results['pred_rel_error'] > 0.3:
        print("""
EXPLANATION: The model has NOT learned to predict EFPs correctly.

When the model fails to learn the non-invariant target, it may instead
learn an invariant approximation (e.g., predicting the mean). This would
explain why symmetry loss still decreases with depth.

RECOMMENDATION: 
- Check training loss - did it converge?
- EFPs may need different hyperparameters (learning rate, epochs, etc.)
- The scale of EFP targets may be very different from KP targets
""")
    elif results['pred_rel_diff'] < 0.05:
        print("""
EXPLANATION: Model predictions ARE Lorentz invariant despite non-invariant targets!

This suggests the model learned an invariant approximation rather than
the true non-invariant function. Possible reasons:
- Optimization finds an invariant local minimum
- The invariant solution has lower loss on average
- Architecture bias toward invariance

RECOMMENDATION:
- Try longer training or different learning rate
- Use data augmentation with Lorentz transforms to force non-invariance
""")
    else:
        print("""
Model appears to be predicting non-invariant EFPs correctly, and
predictions do vary under Lorentz transforms. The decreasing symmetry
loss in intermediate layers suggests the network structure naturally
builds up invariant features before breaking invariance at the output.
""")


if __name__ == '__main__':
    main()

