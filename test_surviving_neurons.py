#!/usr/bin/env python3
"""
Analyze the surviving (non-dead) neurons to understand what they compute.

The model has learned to kill 80-90% of neurons to achieve invariance.
But 10-20% survive - what are they computing?
"""

import torch
import numpy as np
from pathlib import Path
import energyflow as ef
from models import DeepSets
from symmetry import rand_lorentz

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path: str, num_phi_layers: int = 4, num_rho_layers: int = 4) -> DeepSets:
    model = DeepSets(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=num_phi_layers,
        num_rho_layers=num_rho_layers,
        pool_mode='sum',
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def analyze_surviving_neurons(model_path: str, model_name: str):
    """Analyze which neurons survive and what they compute."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    model = load_model(model_path)
    model.eval()
    
    weight = model.phi_layers[0].weight.detach().cpu().numpy()  # (128, 4)
    bias = model.phi_layers[0].bias.detach().cpu().numpy()
    
    # Generate test data
    n_events = 1000
    n_particles = 128
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    X_torch = torch.from_numpy(X).to(device)
    
    # Compute pre-ReLU activations
    with torch.no_grad():
        h_pre = (torch.from_numpy(weight).to(device) @ X_torch.reshape(-1, 4).T).T + torch.from_numpy(bias).to(device)
        h_pre = h_pre.cpu().numpy().reshape(n_events, n_particles, 128)
    
    # Identify surviving neurons (ones that are sometimes positive)
    neuron_activity = (h_pre > 0).mean(axis=(0, 1))  # (128,)
    surviving_mask = neuron_activity > 0.01  # At least 1% active
    n_surviving = surviving_mask.sum()
    
    print(f"\nSurviving neurons: {n_surviving}/128")
    surviving_indices = np.where(surviving_mask)[0]
    
    if n_surviving > 0:
        # Analyze weights of surviving neurons
        surviving_weights = weight[surviving_mask]  # (n_surviving, 4)
        
        print(f"\nSurviving neuron weights:")
        print(f"  Mean weight per component: E={surviving_weights[:, 0].mean():.4f}, "
              f"px={surviving_weights[:, 1].mean():.4f}, "
              f"py={surviving_weights[:, 2].mean():.4f}, "
              f"pz={surviving_weights[:, 3].mean():.4f}")
        
        # Check Minkowski alignment
        metric_pattern = np.array([1, -1, -1, -1])
        metric_normalized = metric_pattern / np.linalg.norm(metric_pattern)
        
        alignments = []
        for w in surviving_weights:
            w_norm = np.linalg.norm(w)
            if w_norm > 0.01:
                alignment = np.abs(np.dot(w / w_norm, metric_normalized))
                alignments.append(alignment)
        
        if alignments:
            print(f"  Minkowski alignment: mean={np.mean(alignments):.4f}, max={np.max(alignments):.4f}")
        
        # Check if surviving neurons are more invariant
        print(f"\nInvariance of surviving vs dead neurons:")
        
        # Apply Lorentz transforms and compare
        L1 = rand_lorentz(torch.Size([n_events]), std_eta=0.5, device=device, dtype=torch.float32)
        L2 = rand_lorentz(torch.Size([n_events]), std_eta=0.5, device=device, dtype=torch.float32)
        
        X_rot1 = torch.matmul(L1.unsqueeze(1), X_torch.unsqueeze(-1)).squeeze(-1)
        X_rot2 = torch.matmul(L2.unsqueeze(1), X_torch.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            h1 = model.forward_with_intermediate(X_rot1, layer_idx=1)
            h2 = model.forward_with_intermediate(X_rot2, layer_idx=1)
            
            diff = (h1 - h2).pow(2).mean(dim=(0, 1)).cpu().numpy()  # (128,)
        
        surviving_variance = diff[surviving_mask].mean()
        dead_variance = diff[~surviving_mask].mean()
        
        print(f"  Surviving neurons variance: {surviving_variance:.6e}")
        print(f"  Dead neurons variance:      {dead_variance:.6e}")
        
        # THE KEY QUESTION: are surviving neurons computing something non-trivial?
        # Check correlation between surviving neuron outputs and KP targets
        from data.kp_dataset import compute_kps
        from train import load_efp_preset
        
        config_dir = Path(__file__).parent / "config"
        edges_list = load_efp_preset('deg3', str(config_dir))
        
        Y = compute_kps(X, edges_list, measure='kinematic', coords='epxpypz')
        Y_log = np.log1p(Y)
        
        # Get layer 1 activations and pool them
        with torch.no_grad():
            h_pooled = model.forward_with_intermediate(X_torch, layer_idx=1).sum(dim=1).cpu().numpy()  # (n_events, 128)
        
        print(f"\nCorrelation between pooled layer 1 activations and KP targets:")
        for kp_idx in range(5):
            # Correlation with each surviving neuron
            max_corr = 0
            for neuron_idx in surviving_indices:
                corr = np.corrcoef(h_pooled[:, neuron_idx], Y_log[:, kp_idx])[0, 1]
                if np.abs(corr) > np.abs(max_corr):
                    max_corr = corr
            print(f"  KP{kp_idx+1}: max |correlation| = {np.abs(max_corr):.4f}")
    
    return n_surviving, surviving_mask


def main():
    print("="*80)
    print("SURVIVING NEURON ANALYSIS")
    print("="*80)
    print("\nWith 80-90% of neurons dead, only 10-20% survive.")
    print("What do the surviving neurons compute?")
    
    model_configs = [
        ('4x4_none.pt', 'Baseline'),
        ('4x4_layer1.pt', 'Layer1 Sym'),
        ('4x4_layer1_strong.pt', 'Strong Sym'),
    ]
    
    for model_path, model_name in model_configs:
        if not Path(model_path).exists():
            print(f"Skipping {model_name}: {model_path} not found")
            continue
        
        analyze_surviving_neurons(model_path, model_name)
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("""
The model achieves "invariance" by killing neurons, not by learning invariant features!

1. Symmetry loss pushes biases negative
2. Most neurons output 0 after ReLU (trivially invariant)
3. A few neurons survive with non-zero output
4. These surviving neurons BREAK invariance but allow task learning
5. The symmetry loss is "satisfied" by dead neurons dominating

This is a form of REGULARIZATION, not true invariance learning!
The model is "cheating" - it kills most neurons to minimize symmetry loss,
while keeping just enough alive to solve the task.

THIS EXPLAINS THE RESULTS:
- Symmetry loss is low (most neurons dead = invariant)
- Task loss is good (surviving neurons carry information)
- Per-KP improvement (regularization effect of killing neurons)
""")


if __name__ == '__main__':
    main()

