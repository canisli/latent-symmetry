"""Visualize Lorentz orbit of particles in the phi representation space.

Loads a trained DeepSets model, generates a Lorentz orbit from 3 fixed particles,
extracts layer 1 phi representations, and plots a 2D PCA visualization.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models import DeepSets
from symmetry import rand_lorentz


def create_fixed_particles(input_scale: float = 0.9515689, seed: int = 42) -> torch.Tensor:
    """Create 3 fixed particles with energy similar to train.py.
    
    Args:
        input_scale: Scale factor matching train.py
        seed: Random seed for reproducibility
        
    Returns:
        Tensor of shape (1, 3, 4) representing [E, px, py, pz]
    """
    np.random.seed(seed)
    
    # Generate random momenta
    px = np.random.randn(3) * input_scale
    py = np.random.randn(3) * input_scale
    pz = np.random.randn(3) * input_scale
    
    # Compute energy (massless particles: E = |p|)
    E = np.sqrt(px**2 + py**2 + pz**2)
    
    # Stack into 4-vectors [E, px, py, pz]
    particles = np.stack([E, px, py, pz], axis=-1).astype(np.float32)
    
    # Add batch dimension: (1, 3, 4)
    return torch.from_numpy(particles).unsqueeze(0)


def generate_lorentz_orbit(
    particles: torch.Tensor,
    num_samples: int,
    std_eta: float = 0.5,
    seed: int = None,
) -> torch.Tensor:
    """Generate Lorentz orbit by applying random transformations to particles.
    
    Args:
        particles: Input particles of shape (1, n_particles, 4)
        num_samples: Number of Lorentz transformations to sample
        std_eta: Rapidity standard deviation for boosts
        seed: Random seed for reproducibility
        
    Returns:
        Tensor of shape (num_samples, n_particles, 4)
    """
    device = particles.device
    dtype = particles.dtype
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Sample random Lorentz transformations: (num_samples, 4, 4)
    L = rand_lorentz(
        shape=torch.Size([num_samples]),
        std_eta=std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    
    # Expand particles to (num_samples, n_particles, 4)
    # particles: (1, n_particles, 4) -> (num_samples, n_particles, 4)
    x = particles.expand(num_samples, -1, -1)
    
    # Apply Lorentz transformations
    # L: (num_samples, 4, 4) -> (num_samples, 1, 4, 4) for broadcasting
    # x: (num_samples, n_particles, 4) -> (num_samples, n_particles, 4, 1)
    # Result: (num_samples, n_particles, 4)
    x_transformed = torch.matmul(L.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    return x_transformed


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Lorentz orbit in phi representation space"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved .pt model file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of Lorentz transformations to sample (default: 500)",
    )
    parser.add_argument(
        "--std-eta",
        type=float,
        default=0.5,
        help="Rapidity std for boosts (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create DeepSets model with same architecture as train.py
    model = DeepSets(
        in_channels=4,
        out_channels=5,
        hidden_channels=128,
        num_phi_layers=4,
        num_rho_layers=4,
        pool_mode='sum',
    ).to(device)
    
    # Load model weights
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from: {args.model_path}")
    print(f"Device: {device}")
    print(f"Sampling {args.num_samples} Lorentz transformations with std_eta={args.std_eta}")
    
    # Create 3 fixed particles
    particles = create_fixed_particles(seed=args.seed).to(device)
    print(f"Created {particles.shape[1]} fixed particles")
    
    # Generate Lorentz orbit
    orbit = generate_lorentz_orbit(
        particles,
        num_samples=args.num_samples,
        std_eta=args.std_eta,
        seed=args.seed + 1,  # Different seed for orbit sampling
    )
    print(f"Generated orbit with shape: {orbit.shape}")
    
    # Extract phi layer 1 activations
    with torch.no_grad():
        phi_activations = model.forward_with_intermediate(orbit, layer_idx=1)
    
    # phi_activations shape: (num_samples, 3, 128)
    print(f"Phi activations shape: {phi_activations.shape}")
    
    # Move to CPU for PCA
    phi_activations = phi_activations.cpu().numpy()
    num_samples, n_particles, hidden_dim = phi_activations.shape
    
    # Reshape to (num_samples * n_particles, hidden_dim) for PCA
    phi_flat = phi_activations.reshape(-1, hidden_dim)
    
    # Perform 2D PCA
    pca = PCA(n_components=2)
    phi_2d = pca.fit_transform(phi_flat)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Reshape back to (num_samples, n_particles, 2)
    phi_2d = phi_2d.reshape(num_samples, n_particles, 2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green
    markers = ['o', 's', '^']  # Circle, Square, Triangle
    
    for particle_idx in range(n_particles):
        x = phi_2d[:, particle_idx, 0]
        y = phi_2d[:, particle_idx, 1]
        ax.scatter(
            x, y,
            c=colors[particle_idx],
            marker=markers[particle_idx],
            alpha=0.6,
            s=30,
            label=f'Particle {particle_idx + 1}',
            edgecolors='white',
            linewidths=0.5,
        )
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(
        f'Lorentz Orbit in Phi Layer 1 Space (PCA)\n'
        f'Model: {args.model_path}\n'
        f'{args.num_samples} samples, std_η={args.std_eta}',
        fontsize=11,
    )
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Add variance explained annotation
    var_text = f'Var explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}'
    ax.annotate(
        var_text,
        xy=(0.02, 0.02),
        xycoords='axes fraction',
        fontsize=9,
        alpha=0.7,
    )
    
    plt.tight_layout()
    
    # Histogram of PC1 for each particle
    fig_hist, axes_hist = plt.subplots(1, n_particles, figsize=(12, 4))
    
    # Compute shared bins across all particles for fair comparison
    all_pc1 = phi_2d[:, :, 0].flatten()
    bins = np.linspace(all_pc1.min(), all_pc1.max(), 40)
    
    for particle_idx in range(n_particles):
        ax = axes_hist[particle_idx]
        pc1_vals = phi_2d[:, particle_idx, 0]
        
        ax.hist(
            pc1_vals,
            bins=bins,
            color=colors[particle_idx],
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
        )
        
        # Add statistics
        mean_val = np.mean(pc1_vals)
        std_val = np.std(pc1_vals)
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'μ={mean_val:.2f}')
        
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Particle {particle_idx + 1}\nσ={std_val:.3f}', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
    
    fig_hist.suptitle(f'PC1 Distribution per Particle\nModel: {args.model_path}', fontsize=11)
    plt.tight_layout()
    
    # Histogram of all 128 hidden dimensions across the Lorentz orbit
    # Flatten all values from all dimensions into a single histogram
    fig_hidden, ax_hidden = plt.subplots(figsize=(10, 6))
    
    # Flatten all values: (num_samples * n_particles * hidden_dim,)
    all_values = phi_flat.flatten()
    
    ax_hidden.hist(
        all_values,
        bins=50,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
    )
    
    # Add mean line
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    ax_hidden.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.4f}')
    ax_hidden.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±1σ = {std_val:.4f}')
    ax_hidden.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax_hidden.set_xlabel('Hidden Dimension Value', fontsize=12)
    ax_hidden.set_ylabel('Count', fontsize=12)
    ax_hidden.set_title(
        f'Distribution of All 128 Hidden Dimensions Across Lorentz Orbit\n'
        f'Model: {args.model_path}\n'
        f'{args.num_samples} samples × {n_particles} particles × {hidden_dim} dimensions = {len(all_values)} total values',
        fontsize=11
    )
    ax_hidden.legend(loc='best')
    ax_hidden.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.show()


if __name__ == "__main__":
    main()

