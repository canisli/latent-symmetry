#!/usr/bin/env python3
"""
Diagnose how Lorentz symmetry emerges through network depth.

For a no-penalty baseline model, evaluates relative symmetry loss at each hidden
layer to test whether symmetry naturally increases with depth.

The relative symmetry loss is:
    ||a - b||^2 / (||a||^2 + ||b||^2 + eps)

This normalization makes losses comparable across layers with different activation scales.
Values are bounded in [0, 2]: 0 = perfect invariance, 2 = opposite directions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from models import DeepSets
from data.kp_dataset import make_kp_dataloader, compute_kps, KPSurrogateDataset
from symmetry import rand_lorentz, lorentz_inverse
from train import load_efp_preset, run_training, derive_seed
import energyflow as ef
import random
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Augmented Dataset with Stored Transformations
# ============================================================================

class AugmentedKPDataset(Dataset):
    """
    Dataset that stores per-sample Lorentz transformations along with data.
    
    X: (N, M, 4) augmented four-momenta (already transformed by G)
    Y: (N, 1, K) KP targets computed from augmented data
    G: (N, 4, 4) Lorentz transformations applied to each event
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, G: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        G = np.asarray(G, dtype=np.float32)
        if Y.ndim == 2:  # (N, K) -> (N, 1, K)
            Y = Y[:, None, :]
        assert X.ndim == 3 and X.shape[-1] == 4, X.shape
        assert Y.ndim == 3, Y.shape
        assert G.ndim == 3 and G.shape[-2:] == (4, 4), G.shape
        assert X.shape[0] == Y.shape[0] == G.shape[0], (X.shape, Y.shape, G.shape)
        self.X = X
        self.Y = Y
        self.G = G

    def __len__(self): 
        return self.X.shape[0]
    
    def __getitem__(self, idx): 
        return self.X[idx], self.Y[idx], self.G[idx]


def make_augmented_kp_dataloader(
    edges_list,
    n_events: int = 10_000,
    n_particles: int = 128,
    batch_size: int = 256,
    measure: str = 'kinematic',
    coords: str = 'epxpypz',
    input_scale: float = 1.0,
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
    target_transform: str = 'log1p',
    std_eta: float = 0.5,
    n_max_std_eta: float = 3.0,
    augmentation_seed: int = None,
):
    """
    Generate a DataLoader with Lorentz-augmented data and stored transformations.
    
    The augmentation workflow:
    1. Generate raw four-momenta X
    2. Sample per-event Lorentz transformations G
    3. Apply augmentation: X_aug = G @ X
    4. Compute labels Y from augmented data X_aug
    5. Return dataset with X_aug, Y, and G
    
    Args:
        edges_list: List of edge configurations for EFPs/KPs
        n_events: Number of events to generate
        n_particles: Number of particles per event
        batch_size: Batch size for DataLoader
        measure: 'kinematic' (Lorentz invariant) or 'eeefm' (non-invariant)
        coords: Coordinate system ('epxpypz')
        input_scale: Scale factor for input four-momenta
        beta: Angular weighting exponent (used by eeefm)
        kappa: Energy weighting exponent (used by eeefm)
        normed: If True, normalize energies
        target_transform: How to transform targets ('log1p', etc.)
        std_eta: Standard deviation of rapidity for augmentation boosts
        n_max_std_eta: Maximum number of standard deviations for truncation
        augmentation_seed: Random seed for augmentation (for reproducibility)
    
    Returns:
        DataLoader yielding (X_aug, Y, G) tuples
    """
    # Generate synthetic events as (E,px,py,pz)
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    # Sample per-event Lorentz transformations
    if augmentation_seed is not None:
        aug_generator = torch.Generator().manual_seed(augmentation_seed)
    else:
        aug_generator = None
    
    G = rand_lorentz(
        shape=torch.Size([n_events]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device='cpu',
        dtype=torch.float32,
        generator=aug_generator,
    ).numpy()  # (N, 4, 4)
    
    # Apply augmentation: X_aug[i] = G[i] @ X[i] for each event
    # X: (N, M, 4) -> need to apply (N, 4, 4) @ (N, M, 4)
    # Result shape: (N, M, 4)
    X_aug = np.einsum('nij,nmj->nmi', G, X)
    
    # Compute KP/EFP targets on AUGMENTED momenta (before input scaling)
    Y = compute_kps(X_aug, edges_list, measure=measure, coords=coords, 
                    beta=beta, kappa=kappa, normed=normed)
    
    # Apply target transformation
    if target_transform == 'log1p':
        Y = np.log1p(Y)
    elif target_transform == 'log_standardized':
        Y = np.log(Y + 1e-10)
        Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-10)
    elif target_transform == 'standardized':
        Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-10)
    else:
        raise ValueError(f"Unknown target_transform: {target_transform}")
    
    # Scale inputs AFTER computing targets
    denom = float(input_scale) if input_scale not in (0, None) else 1.0
    X_aug = X_aug / denom
    
    # Create dataset and dataloader
    dataset = AugmentedKPDataset(X_aug, Y, G)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return dataloader


# ============================================================================
# Relative Symmetry Loss Function
# ============================================================================

def relative_symmetry_loss_augmented(
    model,
    x: torch.Tensor,
    g: torch.Tensor,
    layer_idx: int,
    std_eta: float = 0.5,
    n_max_std_eta: float = 3.0,
    generator: torch.Generator = None,
    mask: torch.Tensor = None,
    eps: float = 1e-8,
):
    """
    Compute RELATIVE symmetry loss with augmentation-aware transformations.
    
    When data has been augmented with transformation g (x_aug = g @ x_orig),
    we use g' @ g^(-1) as the transformation instead of random L.
    This ensures (g' @ g^(-1)) @ (g @ x_orig) = g' @ x_orig, keeping
    the transformed data in the same distribution as the original.
    
    Args:
        model: Model with forward_with_intermediate() method
        x: Input 4-vectors of shape (batch_size, num_particles, 4) - already augmented
        g: Lorentz transformations used for augmentation, shape (batch_size, 4, 4)
        layer_idx: Which layer's activations to compare (-1 for output)
        std_eta: Standard deviation of rapidity for boosts
        n_max_std_eta: Maximum number of standard deviations for truncation
        generator: Optional random generator for reproducibility
        mask: Optional boolean mask of shape (batch_size, num_particles)
        eps: Small constant for numerical stability
    
    Returns:
        Scalar relative symmetry loss
    """
    B, N, D = x.shape
    assert D == 4, f"Expected 4-vectors, got dimension {D}"
    device, dtype = x.device, x.dtype
    
    # Compute g^(-1) for each sample
    g_inv = lorentz_inverse(g)  # (B, 4, 4)
    
    # Sample two fresh Lorentz transformations g1' and g2'
    g1_prime = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    g2_prime = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    
    # Compute compensating transformations: L1 = g1' @ g^(-1), L2 = g2' @ g^(-1)
    L1 = torch.bmm(g1_prime, g_inv)  # (B, 4, 4)
    L2 = torch.bmm(g2_prime, g_inv)  # (B, 4, 4)
    
    # Apply transformations to all particles
    # x: (B, N, 4) -> (B, N, 4, 1)
    # L: (B, 4, 4) -> (B, 1, 4, 4) for broadcasting
    x_rot1 = torch.matmul(L1.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.matmul(L2.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    # Get intermediate activations
    h1 = model.forward_with_intermediate(x_rot1, layer_idx, mask=mask)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx, mask=mask)
    
    # Handle per-particle representations (pre-pooling layers)
    if h1.dim() == 3:  # (B, N, hidden)
        if mask is None:
            mask = torch.any(x != 0.0, dim=-1)  # (B, N)
        
        diff = h1 - h2
        diff_sq = diff.pow(2).sum(dim=-1)  # (B, N)
        h1_sq = h1.pow(2).sum(dim=-1)  # (B, N)
        h2_sq = h2.pow(2).sum(dim=-1)  # (B, N)
        
        per_particle_rel_loss = diff_sq / (h1_sq + h2_sq + eps)  # (B, N)
        
        mask_float = mask.float()
        total_valid = mask_float.sum().clamp(min=1.0)
        loss = (per_particle_rel_loss * mask_float).sum() / total_valid
        
        return loss
    
    # Post-pooling layers: h1 and h2 have shape (B, hidden)
    diff = h1 - h2
    diff_sq = diff.pow(2).sum(dim=-1)  # (B,)
    h1_sq = h1.pow(2).sum(dim=-1)  # (B,)
    h2_sq = h2.pow(2).sum(dim=-1)  # (B,)
    
    per_sample_rel_loss = diff_sq / (h1_sq + h2_sq + eps)  # (B,)
    loss = per_sample_rel_loss.mean()
    
    return loss


def compute_oracle_symmetry_penalty(
    edges_list,
    n_events: int = 1000,
    n_particles: int = 128,
    n_transforms: int = 100,
    measure: str = 'eeefm',
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
    target_transform: str = 'log1p',
    std_eta: float = 0.5,
    n_max_std_eta: float = 3.0,
):
    """
    Compute the relative symmetry penalty for a perfect EFP/KP regressor.
    
    This is the irreducible symmetry loss that a perfect regressor would incur
    when using the augmentation-aware symmetry loss with g' @ g^(-1) transforms.
    
    The penalty is: ||y(L1·x) - y(L2·x)||² / (||y(L1·x)||² + ||y(L2·x)||²)
    
    where L1 = g1' @ g^(-1) and L2 = g2' @ g^(-1) are compensating transforms,
    and y(·) computes the target EFP/KP values.
    
    For Lorentz-invariant targets (kinematic), this penalty should be ~0.
    For non-invariant targets (EFP), this penalty represents the irreducible
    minimum that any model would have when using symmetry loss.
    
    Args:
        edges_list: List of edge configurations for EFPs/KPs
        n_events: Number of events to sample
        n_particles: Particles per event
        n_transforms: Number of transform pairs to sample per event
        measure: 'kinematic' or 'eeefm'
        beta, kappa, normed: EFP measure parameters
        target_transform: How targets are transformed ('log1p', etc.)
        std_eta: Rapidity std for boosts
        n_max_std_eta: Maximum std multiplier
    
    Returns:
        dict with:
            - 'mean_penalty': Mean relative symmetry penalty across events
            - 'std_penalty': Std of penalty across events
            - 'per_kp_penalty': Per-KP penalty array (averaged over events)
    """
    # Generate raw events
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    # Sample initial augmentation transforms G
    G = rand_lorentz(
        shape=torch.Size([n_events]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device='cpu',
        dtype=torch.float32,
    ).numpy()  # (N, 4, 4)
    
    # Apply augmentation: X_aug = G @ X
    X_aug = np.einsum('nij,nmj->nmi', G, X)  # (N, M, 4)
    
    # Convert G to torch for inverse computation
    G_torch = torch.from_numpy(G)
    G_inv = lorentz_inverse(G_torch).numpy()  # (N, 4, 4)
    
    # Collect penalties per event
    event_penalties = []
    per_kp_penalties = []
    
    for event_idx in range(n_events):
        x_aug = X_aug[event_idx:event_idx+1]  # (1, M, 4)
        g_inv = G_inv[event_idx:event_idx+1]  # (1, 4, 4)
        
        transform_penalties = []
        
        for _ in range(n_transforms):
            # Sample two fresh transforms g1' and g2'
            g1_prime = rand_lorentz(
                shape=torch.Size([1]),
                std_eta=std_eta,
                n_max_std_eta=n_max_std_eta,
                device='cpu',
                dtype=torch.float32,
            ).numpy()
            g2_prime = rand_lorentz(
                shape=torch.Size([1]),
                std_eta=std_eta,
                n_max_std_eta=n_max_std_eta,
                device='cpu',
                dtype=torch.float32,
            ).numpy()
            
            # Compute compensating transforms: L1 = g1' @ g^(-1), L2 = g2' @ g^(-1)
            L1 = np.einsum('nij,njk->nik', g1_prime, g_inv)  # (1, 4, 4)
            L2 = np.einsum('nij,njk->nik', g2_prime, g_inv)  # (1, 4, 4)
            
            # Apply transforms to augmented data
            x_rot1 = np.einsum('nij,nmj->nmi', L1, x_aug)  # (1, M, 4)
            x_rot2 = np.einsum('nij,nmj->nmi', L2, x_aug)  # (1, M, 4)
            
            # Compute targets on transformed data
            y1 = compute_kps(x_rot1, edges_list, measure=measure, coords='epxpypz',
                            beta=beta, kappa=kappa, normed=normed)  # (1, K)
            y2 = compute_kps(x_rot2, edges_list, measure=measure, coords='epxpypz',
                            beta=beta, kappa=kappa, normed=normed)  # (1, K)
            
            # Apply target transformation
            if target_transform == 'log1p':
                y1 = np.log1p(y1)
                y2 = np.log1p(y2)
            elif target_transform == 'log_standardized':
                y1 = np.log(y1 + 1e-10)
                y2 = np.log(y2 + 1e-10)
            elif target_transform == 'standardized':
                pass  # No transformation
            
            # Compute relative symmetry penalty per KP
            diff_sq = (y1 - y2) ** 2  # (1, K)
            sum_sq = y1 ** 2 + y2 ** 2  # (1, K)
            per_kp_rel_loss = diff_sq / (sum_sq + 1e-8)  # (1, K)
            
            transform_penalties.append(per_kp_rel_loss[0])  # (K,)
        
        # Average over transforms for this event
        event_per_kp = np.mean(transform_penalties, axis=0)  # (K,)
        per_kp_penalties.append(event_per_kp)
        event_penalties.append(np.mean(event_per_kp))  # scalar
    
    # Average over events
    mean_penalty = np.mean(event_penalties)
    std_penalty = np.std(event_penalties) / np.sqrt(n_events)
    per_kp_penalty = np.mean(per_kp_penalties, axis=0)  # (K,)
    
    return {
        'mean_penalty': mean_penalty,
        'std_penalty': std_penalty,
        'per_kp_penalty': per_kp_penalty,
    }


def relative_symmetry_loss(
    model,
    x: torch.Tensor,
    layer_idx: int,
    std_eta: float = 0.5,
    n_max_std_eta: float = 3.0,
    generator: torch.Generator = None,
    mask: torch.Tensor = None,
    eps: float = 1e-8,
):
    """
    Compute RELATIVE symmetry loss at a given layer.
    
    The relative loss is: ||a - b||^2 / (||a||^2 + ||b||^2 + eps)
    
    This makes losses comparable across layers since it normalizes by activation scale.
    Values are bounded in [0, 2]: 0 = perfect invariance.
    
    Args:
        model: Model with forward_with_intermediate() method
        x: Input 4-vectors of shape (batch_size, num_particles, 4)
        layer_idx: Which layer's activations to compare (-1 for output)
        std_eta: Standard deviation of rapidity for boosts
        n_max_std_eta: Maximum number of standard deviations for truncation
        generator: Optional random generator for reproducibility
        mask: Optional boolean mask of shape (batch_size, num_particles)
        eps: Small constant for numerical stability
    
    Returns:
        Scalar relative symmetry loss
    """
    B, N, D = x.shape
    assert D == 4, f"Expected 4-vectors, got dimension {D}"
    device, dtype = x.device, x.dtype
    
    # Sample two random Lorentz transformations
    L1 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    L2 = rand_lorentz(
        shape=torch.Size([B]),
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    
    # Apply Lorentz transformations to all particles
    x_rot1 = torch.matmul(L1.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.matmul(L2.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    # Get intermediate activations
    h1 = model.forward_with_intermediate(x_rot1, layer_idx, mask=mask)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx, mask=mask)
    
    # Handle per-particle representations (pre-pooling layers)
    if h1.dim() == 3:  # (B, N, hidden)
        if mask is None:
            mask = torch.any(x != 0.0, dim=-1)  # (B, N)
        
        # Compute squared norms and differences per particle
        # h1, h2: (B, N, hidden)
        diff = h1 - h2
        diff_sq = diff.pow(2).sum(dim=-1)  # (B, N)
        h1_sq = h1.pow(2).sum(dim=-1)  # (B, N)
        h2_sq = h2.pow(2).sum(dim=-1)  # (B, N)
        
        # Relative loss per particle: ||a-b||^2 / (||a||^2 + ||b||^2 + eps)
        per_particle_rel_loss = diff_sq / (h1_sq + h2_sq + eps)  # (B, N)
        
        # Average over valid particles
        mask_float = mask.float()
        total_valid = mask_float.sum().clamp(min=1.0)
        loss = (per_particle_rel_loss * mask_float).sum() / total_valid
        
        return loss
    
    # Post-pooling layers: h1 and h2 have shape (B, hidden)
    diff = h1 - h2
    diff_sq = diff.pow(2).sum(dim=-1)  # (B,)
    h1_sq = h1.pow(2).sum(dim=-1)  # (B,)
    h2_sq = h2.pow(2).sum(dim=-1)  # (B,)
    
    # Relative loss per sample
    per_sample_rel_loss = diff_sq / (h1_sq + h2_sq + eps)  # (B,)
    loss = per_sample_rel_loss.mean()
    
    return loss


# ============================================================================
# Diagnostic Functions
# ============================================================================

def evaluate_layer_symmetry(
    model,
    dataloader,
    layer_idx: int,
    std_eta: float = 0.5,
    n_samples: int = 5,
    device: str = 'cuda',
):
    """
    Evaluate relative symmetry loss at a specific layer over the dataset.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        layer_idx: Layer index to evaluate
        std_eta: Rapidity std for Lorentz transforms
        n_samples: Number of random transform pairs to sample per batch
        device: Device to use
    
    Returns:
        mean_loss: Mean relative symmetry loss
        std_loss: Standard deviation across batches
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            
            # Sample multiple transform pairs for more robust estimate
            batch_losses = []
            for _ in range(n_samples):
                loss = relative_symmetry_loss(
                    model, xb, layer_idx, std_eta=std_eta
                )
                batch_losses.append(loss.item())
            
            losses.append(np.mean(batch_losses))
    
    return np.mean(losses), np.std(losses) / np.sqrt(len(losses))


def get_layer_names(num_phi_layers: int, num_rho_layers: int):
    """Generate descriptive names for each layer index."""
    names = {}
    
    # Phi layers (per-particle)
    for i in range(1, num_phi_layers + 1):
        names[i] = f"phi_{i}"
    
    # Pooling layer
    pool_idx = num_phi_layers + 1
    names[pool_idx] = "pool"
    
    # Rho layers (post-pooling)
    for i in range(1, num_rho_layers + 1):
        rho_idx = pool_idx + i
        names[rho_idx] = f"rho_{i}"
    
    # Output
    names[-1] = "output"
    
    return names


def diagnose_model(
    model_path: str,
    num_events: int = 2000,
    n_particles: int = 128,
    batch_size: int = 256,
    input_scale: float = 0.9515689,
    std_eta: float = 0.5,
    n_samples: int = 5,
    device: str = None,
    target_type: str = 'kinematic',
    efp_preset: str = 'deg3',
    efp_beta: float = 2.0,
    efp_kappa: float = 1.0,
    efp_normed: bool = False,
    target_transform: str = 'log1p',
):
    """
    Run full layer-wise symmetry diagnosis on a trained model.
    
    Args:
        model_path: Path to saved model weights
        num_events: Number of events to evaluate on
        n_particles: Particles per event
        batch_size: Batch size for evaluation
        input_scale: Data input scale
        std_eta: Rapidity std for Lorentz transforms
        n_samples: Transform samples per batch
        device: Device to use (auto-detect if None)
        target_type: 'kinematic' (Lorentz invariant) or 'efp' (non-invariant)
        efp_preset: EFP preset name (must match training)
        efp_beta: Angular weighting for EFP measure (only used if target_type='efp')
        efp_kappa: Energy weighting for EFP measure (only used if target_type='efp')
        efp_normed: Whether to normalize energies for EFP (only used if target_type='efp')
        target_transform: How targets are transformed ('log1p', 'log_standardized', 'standardized')
    
    Returns:
        results: Dict mapping layer_idx -> (mean_loss, std_err, layer_name)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model architecture (matching 4x4 models)
    num_phi_layers = 4
    num_rho_layers = 4
    hidden_channels = 128
    
    # Load EFP preset to determine output channels
    edges_list = load_efp_preset(efp_preset, 'config')
    num_kps = len(edges_list)
    
    # Create model
    model = DeepSets(
        in_channels=4,
        out_channels=num_kps,
        hidden_channels=hidden_channels,
        num_phi_layers=num_phi_layers,
        num_rho_layers=num_rho_layers,
        pool_mode='sum',
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Determine measure parameters based on target type
    if target_type == 'kinematic':
        measure = 'kinematic'
        beta = 2.0  # ignored by kinematic measure
        kappa = 1.0  # ignored by kinematic measure
        normed = False  # ignored by kinematic measure
        print(f"Target type: kinematic (Lorentz invariant)")
    elif target_type == 'efp':
        measure = 'eeefm'
        beta = efp_beta
        kappa = efp_kappa
        normed = efp_normed
        normed_str = "normed" if normed else "unnormed"
        print(f"Target type: EFP (eeefm, beta={beta}, kappa={kappa}, {normed_str}) - NOT Lorentz invariant")
    else:
        raise ValueError(f"Unknown target_type: {target_type}. Must be 'kinematic' or 'efp'")
    
    # Create test dataloader with appropriate measure
    dataloader = make_kp_dataloader(
        edges_list=edges_list,
        n_events=num_events,
        n_particles=n_particles,
        batch_size=batch_size,
        input_scale=input_scale,
        measure=measure,
        beta=beta,
        kappa=kappa,
        normed=normed,
        target_transform=target_transform,
    )
    
    # Get layer names
    layer_names = get_layer_names(num_phi_layers, num_rho_layers)
    
    # Evaluate each layer
    results = {}
    max_layer_idx = num_phi_layers + num_rho_layers + 1
    
    # All layer indices including output
    layer_indices = list(range(1, max_layer_idx + 1)) + [-1]
    
    print(f"\nEvaluating relative symmetry at {len(layer_indices)} layers...")
    print(f"  std_eta = {std_eta}, n_samples = {n_samples}")
    print(f"  {num_events} events, {n_particles} particles/event\n")
    
    for layer_idx in tqdm(layer_indices, desc="Layers"):
        mean_loss, std_err = evaluate_layer_symmetry(
            model, dataloader, layer_idx,
            std_eta=std_eta, n_samples=n_samples, device=device,
        )
        name = layer_names.get(layer_idx, f"layer_{layer_idx}")
        results[layer_idx] = (mean_loss, std_err, name)
    
    # Compute oracle baseline for EFP targets (non-invariant)
    # This is the irreducible symmetry penalty a perfect regressor would incur
    oracle_baseline = None
    if target_type == 'efp':
        print("\nComputing oracle baseline (perfect EFP regressor)...")
        oracle_baseline = compute_oracle_symmetry_penalty(
            edges_list=edges_list,
            n_events=min(100, num_events),  # Use fewer events for speed
            n_particles=n_particles,
            n_transforms=50,
            measure=measure,
            beta=beta,
            kappa=kappa,
            normed=normed,
            target_transform=target_transform,
            std_eta=std_eta,
        )
        print(f"  Oracle penalty: {oracle_baseline['mean_penalty']:.6f} ± {oracle_baseline['std_penalty']:.6f}")
    
    results['oracle_baseline'] = oracle_baseline
    results['target_type'] = target_type
    
    return results


def print_results(results: dict):
    """Print results table."""
    print("\n" + "=" * 60)
    print("LAYER-WISE RELATIVE SYMMETRY LOSS")
    print("=" * 60)
    print(f"{'Layer':<12} {'Index':<8} {'Rel. Sym Loss':<18} {'Std Err':<12}")
    print("-" * 60)
    
    # Sort by layer index (but put -1 at end), excluding metadata keys
    layer_keys = [k for k in results.keys() if isinstance(k, int) and k > 0]
    sorted_keys = sorted(layer_keys) + [-1]
    
    for layer_idx in sorted_keys:
        mean_loss, std_err, name = results[layer_idx]
        idx_str = str(layer_idx) if layer_idx > 0 else "output"
        print(f"{name:<12} {idx_str:<8} {mean_loss:<18.6f} {std_err:<12.6f}")
    
    print("=" * 60)
    
    # Oracle baseline for EFP targets
    oracle_baseline = results.get('oracle_baseline')
    target_type = results.get('target_type', 'kinematic')
    if oracle_baseline is not None:
        print("\nORACLE BASELINE (Perfect EFP Regressor):")
        print(f"  A perfect EFP regressor would incur this irreducible symmetry penalty")
        print(f"  because EFPs are NOT Lorentz invariant.")
        print(f"  Mean penalty: {oracle_baseline['mean_penalty']:.6f} ± {oracle_baseline['std_penalty']:.6f}")
        print("-" * 60)
    
    # Analysis
    phi_losses = [results[i][0] for i in sorted_keys if results[i][2].startswith('phi')]
    rho_losses = [results[i][0] for i in sorted_keys if results[i][2].startswith('rho')]
    pool_loss = results[sorted_keys[len(phi_losses)]][0]  # Pool layer
    output_loss = results[-1][0]
    
    print("\nSUMMARY:")
    print(f"  Phi layers (per-particle): {np.mean(phi_losses):.6f} avg")
    print(f"  Pool layer:                {pool_loss:.6f}")
    print(f"  Rho layers (post-pool):    {np.mean(rho_losses):.6f} avg")
    print(f"  Output:                    {output_loss:.6f}")
    
    # Show comparison to oracle baseline if available
    if oracle_baseline is not None:
        oracle_penalty = oracle_baseline['mean_penalty']
        excess_loss = output_loss - oracle_penalty
        print(f"\n  Oracle baseline:           {oracle_penalty:.6f}")
        print(f"  Excess over oracle:        {excess_loss:.6f} ({excess_loss/oracle_penalty*100:.1f}% above baseline)")
    
    # Trend analysis
    all_losses = phi_losses + [pool_loss] + rho_losses + [output_loss]
    if all_losses[0] > all_losses[-1]:
        print("\n  → Symmetry INCREASES with depth (loss decreases)")
    else:
        print("\n  → Symmetry DECREASES with depth (loss increases)")


def plot_results(results: dict, save_path: str = None):
    """Plot relative symmetry vs layer depth."""
    
    # Sort by layer index, excluding metadata keys
    layer_keys = [k for k in results.keys() if isinstance(k, int) and k > 0]
    sorted_keys = sorted(layer_keys) + [-1]
    
    names = [results[k][2] for k in sorted_keys]
    means = [results[k][0] for k in sorted_keys]
    stds = [results[k][1] for k in sorted_keys]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, 
                  color=['#3498db'] * 4 + ['#2ecc71'] + ['#e74c3c'] * 4 + ['#9b59b6'],
                  edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Relative Symmetry Loss', fontsize=12)
    ax.set_title('Lorentz Symmetry Emergence Through Network Depth\n(Lower = More Invariant)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Phi (per-particle)'),
        Patch(facecolor='#2ecc71', label='Pool'),
        Patch(facecolor='#e74c3c', label='Rho (post-pool)'),
        Patch(facecolor='#9b59b6', label='Output'),
    ]
    
    # Add oracle baseline horizontal line for EFP targets
    oracle_baseline = results.get('oracle_baseline')
    if oracle_baseline is not None:
        oracle_penalty = oracle_baseline['mean_penalty']
        ax.axhline(y=oracle_penalty, color='orange', linestyle='--', linewidth=2, 
                   label=f'Oracle baseline ({oracle_penalty:.4f})')
        legend_elements.append(
            plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, 
                       label=f'Oracle baseline ({oracle_penalty:.4f})')
        )
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_augmented(
    model,
    loss_fn,
    loader,
    optimizer,
    scheduler=None,
    grad_clip=None,
    symmetry_layer=None,
    lambda_sym=0.0,
    std_eta=0.5,
    augmentation_generator=None,
    device='cuda',
):
    """
    Training loop for augmented data with compensating symmetry loss.
    
    The loader yields (X_aug, Y, G) where:
    - X_aug: augmented inputs (already transformed by G)
    - Y: targets computed from augmented data
    - G: per-sample Lorentz transformations
    
    The symmetry loss uses g' @ g^(-1) to ensure transformed data stays in distribution.
    """
    task_batch_losses = []
    sym_batch_losses = []
    
    model.train()
    for batch in loader:
        xb, yb, gb = batch
        xb = xb.to(device)  # (batch_size, num_particles, 4)
        yb = yb.to(device)  # (batch_size, 1, num_kps)
        gb = gb.to(device)  # (batch_size, 4, 4)
        
        # Squeeze the singleton dimension from yb
        yb = yb.squeeze(1)

        optimizer.zero_grad()
        pred = model(xb)
        task_loss = loss_fn(pred, yb)
        
        # Compute augmentation-aware symmetry loss
        sym_loss = torch.tensor(0.0, device=device)
        if symmetry_layer is not None and lambda_sym > 0:
            sym_loss = relative_symmetry_loss_augmented(
                model, xb, gb, symmetry_layer,
                std_eta=std_eta,
                generator=augmentation_generator,
            )
        
        task_batch_losses.append(task_loss.item())
        sym_batch_losses.append(sym_loss.item())
        
        total_loss = task_loss + lambda_sym * sym_loss
        total_loss.backward()
        
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
    
    avg_task_loss = sum(task_batch_losses) / len(task_batch_losses)
    avg_sym_loss = sum(sym_batch_losses) / len(sym_batch_losses) if sym_batch_losses else 0.0
    
    return avg_task_loss, avg_sym_loss


def evaluate_augmented(
    model,
    loss_fn,
    loader,
    symmetry_layer=None,
    std_eta=0.5,
    augmentation_generator=None,
    device='cuda',
):
    """Evaluation loop for augmented data."""
    model.eval()
    task_losses = []
    sym_losses = []
    
    with torch.no_grad():
        for batch in loader:
            xb, yb, gb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            gb = gb.to(device)
            yb = yb.squeeze(1)
            
            pred = model(xb)
            task_losses.append(loss_fn(pred, yb).item())
            
            if symmetry_layer is not None:
                sym_loss = relative_symmetry_loss_augmented(
                    model, xb, gb, symmetry_layer,
                    std_eta=std_eta,
                    generator=augmentation_generator,
                )
                sym_losses.append(sym_loss.item())
    
    avg_task_loss = sum(task_losses) / len(task_losses)
    avg_sym_loss = sum(sym_losses) / len(sym_losses) if sym_losses else 0.0
    
    return avg_task_loss, avg_sym_loss


def run_augmented_training(
    # Data params
    num_events: int = 10000,
    n_particles: int = 128,
    batch_size: int = 256,
    input_scale: float = 0.9515689,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    edges_list=None,
    # Target type params
    target_type: str = 'kinematic',
    efp_measure: str = 'eeefm',
    efp_beta: float = 2.0,
    efp_kappa: float = 1.0,
    efp_normed: bool = False,
    target_transform: str = 'log1p',
    # Training params
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    warmup_epochs: int = 5,
    weight_decay: float = 0.0,
    grad_clip: float = None,
    early_stopping_patience: int = None,
    # Model params
    hidden_channels: int = 128,
    num_phi_layers: int = 4,
    num_rho_layers: int = 4,
    # Symmetry params
    symmetry_enabled: bool = False,
    symmetry_layer: int = None,
    lambda_sym_max: float = 1.0,
    std_eta: float = 0.5,
    # Augmentation params
    augment_std_eta: float = 0.5,
    # Other
    run_seed: int = 42,
    headless: bool = False,
    save_model_path: str = None,
):
    """
    Training function with Lorentz augmentation support.
    
    When augmentation is enabled:
    1. Data is augmented at generation time: X_aug = G @ X
    2. Labels are computed from augmented data
    3. Symmetry loss uses compensating transformations: g' @ g^(-1)
    
    This ensures the symmetry loss samples transformations that keep the
    data in the same distribution as the original.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Derive seeds
    data_seed = derive_seed(run_seed, "data")
    model_seed = derive_seed(run_seed, "model")
    augmentation_seed = derive_seed(run_seed, "augmentation")
    
    # Set seeds
    np.random.seed(data_seed)
    random.seed(data_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    
    # Determine measure parameters based on target type
    if target_type == 'kinematic':
        measure = 'kinematic'
        beta, kappa, normed = 2.0, 1.0, False
    elif target_type == 'efp':
        measure = efp_measure
        beta, kappa, normed = efp_beta, efp_kappa, efp_normed
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    # Build augmented dataloaders
    train_loader = make_augmented_kp_dataloader(
        edges_list=edges_list,
        n_events=int(num_events * train_split),
        n_particles=n_particles,
        batch_size=batch_size,
        measure=measure,
        input_scale=input_scale,
        beta=beta, kappa=kappa, normed=normed,
        target_transform=target_transform,
        std_eta=augment_std_eta,
        augmentation_seed=augmentation_seed,
    )
    val_loader = make_augmented_kp_dataloader(
        edges_list=edges_list,
        n_events=int(num_events * val_split),
        n_particles=n_particles,
        batch_size=batch_size,
        measure=measure,
        input_scale=input_scale,
        beta=beta, kappa=kappa, normed=normed,
        target_transform=target_transform,
        std_eta=augment_std_eta,
        augmentation_seed=augmentation_seed + 1,  # Different seed for val
    )
    test_loader = make_augmented_kp_dataloader(
        edges_list=edges_list,
        n_events=int(num_events * test_split),
        n_particles=n_particles,
        batch_size=batch_size,
        measure=measure,
        input_scale=input_scale,
        beta=beta, kappa=kappa, normed=normed,
        target_transform=target_transform,
        std_eta=augment_std_eta,
        augmentation_seed=augmentation_seed + 2,  # Different seed for test
    )
    
    # Create model
    num_kps = len(edges_list)
    model = DeepSets(
        in_channels=4,
        out_channels=num_kps,
        hidden_channels=hidden_channels,
        num_phi_layers=num_phi_layers,
        num_rho_layers=num_rho_layers,
        pool_mode='sum',
    ).to(device)
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create augmentation generator for symmetry loss
    aug_generator = torch.Generator(device=device).manual_seed(augmentation_seed)
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    lambda_sym = lambda_sym_max if symmetry_enabled else 0.0
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    pbar = tqdm(range(num_epochs), disable=headless)
    for epoch in pbar:
        train_task_loss, train_sym_loss = train_epoch_augmented(
            model, loss_fn, train_loader, optimizer, scheduler, grad_clip,
            symmetry_layer=symmetry_layer,
            lambda_sym=lambda_sym,
            std_eta=std_eta,
            augmentation_generator=aug_generator,
            device=device,
        )
        val_task_loss, val_sym_loss = evaluate_augmented(
            model, loss_fn, val_loader,
            symmetry_layer=symmetry_layer,
            std_eta=std_eta,
            augmentation_generator=aug_generator,
            device=device,
        )
        
        if val_task_loss < best_val_loss:
            best_val_loss = val_task_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        postfix = {'train': f'{train_task_loss:.4f}', 'val': f'{val_task_loss:.4f}'}
        if symmetry_enabled:
            postfix['sym'] = f'{train_sym_loss:.2e}'
        pbar.set_postfix(postfix)
        
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            if not headless:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    # Save model
    if save_model_path:
        Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        if not headless:
            print(f"\nModel saved to {save_model_path}")
    
    # Final evaluation
    test_task_loss, test_sym_loss = evaluate_augmented(
        model, loss_fn, test_loader,
        symmetry_layer=symmetry_layer,
        std_eta=std_eta,
        augmentation_generator=aug_generator,
        device=device,
    )
    
    if not headless:
        print(f"\n{'='*40}")
        print("AUGMENTED TRAINING RESULTS")
        print(f"{'='*40}")
        print(f"Test task loss: {test_task_loss:.4e}")
        if symmetry_enabled:
            print(f"Test symmetry loss: {test_sym_loss:.4e}")
        print(f"Augmentation std_eta: {augment_std_eta}")
        print(f"{'='*40}")
    
    return {
        'test_task_loss': test_task_loss,
        'test_sym_loss': test_sym_loss if symmetry_enabled else None,
        'augmented': True,
        'augment_std_eta': augment_std_eta,
    }


def train_model_if_needed(
    model_path: str,
    target_type: str = 'kinematic',
    efp_preset: str = 'deg3',
    efp_measure: str = 'eeefm',
    efp_beta: float = 2.0,
    efp_kappa: float = 1.0,
    efp_normed: bool = False,
    target_transform: str = 'log1p',
    num_events: int = 10000,
    n_particles: int = 128,
    batch_size: int = 256,
    input_scale: float = None,  # If None, compute from training data std
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    warmup_epochs: int = 5,
    weight_decay: float = 0.0,
    grad_clip: float = None,
    early_stopping_patience: int = 10,
    num_phi_layers: int = 4,
    num_rho_layers: int = 4,
    hidden_channels: int = 128,
    run_seed: int = 42,
    # Augmentation params
    augment_inputs: bool = False,
    augment_std_eta: float = 0.5,
    # Symmetry params (when training with augmentation)
    symmetry_enabled: bool = False,
    symmetry_layer: int = None,
    lambda_sym_max: float = 1.0,
    std_eta: float = 0.5,
):
    """
    Train a model if it doesn't exist.
    
    Training Target Details:
        The model trains on TRANSFORMED targets, not raw KP/EFP values:
        
        1. Raw targets are computed: KP/EFP values from four-momenta
        2. Targets are transformed (default: 'log1p' = log(1+x)):
           - 'log1p': log(1+x) - compresses large values, good for KPs
           - 'log_standardized': log(x) then z-score normalize
           - 'standardized': z-score normalize raw values
        3. Model learns to predict these transformed targets directly
        4. Loss (MSE) is computed between model outputs and transformed targets
        
        The model NEVER sees raw KP/EFP values - it only sees and predicts
        transformed values. This is important because:
        - Raw KP values can span many orders of magnitude
        - log1p transformation compresses the range and stabilizes training
        - Model outputs are in transformed space (e.g., log1p space)
        
        When evaluating, predictions are compared to transformed targets,
        so metrics (RMSE, etc.) are in transformed space.
    
    Returns:
        Path to the model file (existing or newly trained)
    """
    model_path_obj = Path(model_path)
    
    # If model exists, return early
    if model_path_obj.exists():
        print(f"Model already exists at {model_path}, skipping training.")
        return model_path
    
    # Load EFP preset (needed for both printing and training)
    edges_list = load_efp_preset(efp_preset, 'config')
    
    # Compute input scale from actual training data if not provided
    if input_scale is None:
        # Generate the actual training data that will be used for training
        # Use the same seed derivation as training for consistency
        data_seed = derive_seed(run_seed, "data")
        np.random.seed(data_seed)
        random.seed(data_seed)
        
        # Generate the full training dataset (with train_split applied)
        train_split = 0.6  # Default train split from config
        train_events = int(num_events * train_split)
        X_train = ef.gen_random_events_mcom(train_events, n_particles, dim=4).astype(np.float32)
        
        # Compute input scale as std of the actual training data
        computed_input_scale = float(np.std(X_train))
        if computed_input_scale < 1e-12:
            computed_input_scale = 1.0
        input_scale = computed_input_scale
        input_scale_source = f"computed from actual training data std ({computed_input_scale:.6e}, n={train_events:,} events)"
    else:
        input_scale_source = f"provided ({input_scale:.6e})"
    
    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")
    print(f"Model will be saved to: {model_path}")
    print(f"Architecture: DeepSets ({num_phi_layers} phi, {num_rho_layers} rho layers)")
    print(f"Hidden channels: {hidden_channels}")
    print()
    print("TARGET CONFIGURATION:")
    print(f"  Target type: {target_type}")
    if target_type == 'kinematic':
        print(f"    → Lorentz-invariant kinematic polynomials")
    elif target_type == 'efp':
        print(f"    → Non-invariant EFPs (measure: {efp_measure})")
        print(f"    → EFP parameters: beta={efp_beta}, kappa={efp_kappa}, normed={efp_normed}")
    print(f"  EFP preset: {efp_preset} ({len(edges_list)} polynomials)")
    print()
    print("TARGET TRANSFORMATION:")
    if target_transform == 'log1p':
        print(f"  Transform: log1p (log(1+x))")
        print(f"    → Compresses large values, stabilizes training")
        print(f"    → Model trains on log1p-transformed targets")
        print(f"    → Model outputs are in log1p space")
    elif target_transform == 'log_standardized':
        print(f"  Transform: log_standardized (log(x) then z-score)")
        print(f"    → Preserves relative differences")
        print(f"    → Model trains on log-standardized targets")
    elif target_transform == 'standardized':
        print(f"  Transform: standardized (z-score normalization)")
        print(f"    → Preserves linear structure")
        print(f"    → Model trains on standardized targets")
    print(f"  Note: Model NEVER sees raw KP/EFP values - only transformed values")
    print()
    print("TRAINING PARAMETERS:")
    print(f"  Training events: {num_events:,}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Batch size: {batch_size}")
    print(f"  Particles per event: {n_particles}")
    print(f"  Input scale: {input_scale_source}")
    print(f"  Random seed: {run_seed}")
    if augment_inputs:
        print()
        print("AUGMENTATION:")
        print(f"  Augment inputs: ENABLED")
        print(f"  Augment std_eta: {augment_std_eta}")
        print(f"    → Labels computed from Lorentz-augmented data")
        print(f"    → Symmetry loss uses g'·g^(-1) to stay in-distribution")
        if symmetry_enabled:
            print(f"  Symmetry loss enabled:")
            print(f"    → Layer: {symmetry_layer}")
            print(f"    → Lambda: {lambda_sym_max}")
            print(f"    → Std_eta: {std_eta}")
        else:
            print(f'Symmetry loss disabled during training')
    print(f"{'='*60}\n")
    
    # Train the model (with or without augmentation)
    if augment_inputs:
        # Use augmented training with compensating symmetry loss
        result = run_augmented_training(
            # Data params
            num_events=num_events,
            n_particles=n_particles,
            batch_size=batch_size,
            input_scale=input_scale,
            edges_list=edges_list,
            # Target type params
            target_type=target_type,
            efp_measure=efp_measure,
            efp_beta=efp_beta,
            efp_kappa=efp_kappa,
            efp_normed=efp_normed,
            target_transform=target_transform,
            # Training params
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            early_stopping_patience=early_stopping_patience,
            # Model params
            hidden_channels=hidden_channels,
            num_phi_layers=num_phi_layers,
            num_rho_layers=num_rho_layers,
            # Symmetry params
            symmetry_enabled=symmetry_enabled,
            symmetry_layer=symmetry_layer,
            lambda_sym_max=lambda_sym_max,
            std_eta=std_eta,
            # Augmentation params
            augment_std_eta=augment_std_eta,
            # Other
            run_seed=run_seed,
            headless=False,
            save_model_path=model_path,
        )
    else:
        # Standard training without augmentation
        result = run_training(
            # Data params
            num_events=num_events,
            n_particles=n_particles,
            batch_size=batch_size,
            input_scale=input_scale,
            edges_list=edges_list,
            # Target type params
            target_type=target_type,
            efp_measure=efp_measure,
            efp_beta=efp_beta,
            efp_kappa=efp_kappa,
            efp_normed=efp_normed,
            target_transform=target_transform,
            # Training params
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            early_stopping_patience=early_stopping_patience,
            # Model params
            model_type='deepsets',
            hidden_channels=hidden_channels,
            num_phi_layers=num_phi_layers,
            num_rho_layers=num_rho_layers,
            pool_mode='sum',
            # Symmetry params (disabled for baseline)
            symmetry_enabled=False,
            symmetry_layer=None,
            lambda_sym_max=0.0,
            std_eta=0.5,
            # Other
            run_seed=run_seed,
            headless=False,
            save_model_path=model_path,
        )
    
    print(f"\nTraining completed. Model saved to {model_path}")
    return model_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose layer-wise symmetry emergence')
    parser.add_argument('--model', type=str, default='4x4_none.pt',
                        help='Path to trained model weights (will train if not found)')
    parser.add_argument('--num-events', type=int, default=2000,
                        help='Number of events for evaluation')
    parser.add_argument('--std-eta', type=float, default=0.5,
                        help='Rapidity std for Lorentz transforms')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of transform samples per batch')
    parser.add_argument('--save-plot', type=str, default='layer_symmetry.png',
                        help='Path to save plot')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    # Target type arguments
    parser.add_argument('--target-type', type=str, default='kinematic',
                        choices=['kinematic', 'efp'],
                        help='Target type: kinematic (Lorentz invariant) or efp (non-invariant)')
    parser.add_argument('--efp-preset', type=str, default='deg3',
                        help='EFP preset name (see config/efp_presets.yaml)')
    parser.add_argument('--efp-measure', type=str, default='eeefm',
                        help='EFP measure (only used with --target-type efp)')
    parser.add_argument('--efp-beta', type=float, default=2.0,
                        help='EFP angular weighting exponent (only used with --target-type efp)')
    parser.add_argument('--efp-kappa', type=float, default=1.0,
                        help='EFP energy weighting exponent (only used with --target-type efp)')
    parser.add_argument('--efp-normed', action='store_true',
                        help='Normalize energies for EFP (default: False)')
    parser.add_argument('--target-transform', type=str, default='log1p',
                        choices=['log1p', 'log_standardized', 'standardized'],
                        help='Target transformation (must match training)')
    # Training arguments (only used if model doesn't exist)
    parser.add_argument('--train-num-events', type=int, default=10000,
                        help='Number of events for training (if model needs to be trained)')
    parser.add_argument('--train-num-epochs', type=int, default=100,
                        help='Number of training epochs (if model needs to be trained)')
    parser.add_argument('--train-learning-rate', type=float, default=0.001,
                        help='Learning rate for training (if model needs to be trained)')
    parser.add_argument('--train-early-stopping', type=int, default=10,
                        help='Early stopping patience (if model needs to be trained)')
    parser.add_argument('--train-input-scale', type=float, default=None,
                        help='Input scale for training (if None, compute from training data std)')
    parser.add_argument('--no-train', action='store_true',
                        help='Do not train model if missing (raise error instead)')
    # Augmentation arguments
    parser.add_argument('--augment-inputs', action='store_true',
                        help='Apply Lorentz augmentation to inputs before computing labels. '
                             'The symmetry loss uses g\'·g^(-1) to keep shifts in-distribution.')
    parser.add_argument('--augment-std-eta', type=float, default=0.5,
                        help='Rapidity std for input augmentation (if --augment-inputs)')
    # Symmetry loss arguments (for augmented training)
    parser.add_argument('--symmetry-enabled', action='store_true',
                        help='Enable symmetry loss during training (only with --augment-inputs)')
    parser.add_argument('--symmetry-layer', type=int, default=5,
                        help='Layer index for symmetry loss (default: 5 = pool layer for 4x4)')
    parser.add_argument('--lambda-sym', type=float, default=1.0,
                        help='Weight for symmetry loss')
    
    args = parser.parse_args()
    
    # Check if model exists, train if needed
    if not Path(args.model).exists():
        if args.no_train:
            print(f"Error: Model file not found: {args.model}")
            print("\nAvailable models:")
            for p in Path('.').glob('*.pt'):
                print(f"  {p}")
            return
        
        # Train the model automatically
        train_model_if_needed(
            model_path=args.model,
            target_type=args.target_type,
            efp_preset=args.efp_preset,
            efp_measure=args.efp_measure,
            efp_beta=args.efp_beta,
            efp_kappa=args.efp_kappa,
            efp_normed=args.efp_normed,
            target_transform=args.target_transform,
            num_events=args.train_num_events,
            num_epochs=args.train_num_epochs,
            learning_rate=args.train_learning_rate,
            early_stopping_patience=args.train_early_stopping,
            input_scale=args.train_input_scale,
            # Augmentation params
            augment_inputs=args.augment_inputs,
            augment_std_eta=args.augment_std_eta,
            # Symmetry params (only used with augmentation)
            symmetry_enabled=args.symmetry_enabled,
            symmetry_layer=args.symmetry_layer,
            lambda_sym_max=args.lambda_sym,
            std_eta=args.std_eta,
        )
    
    # Run diagnosis
    results = diagnose_model(
        model_path=args.model,
        num_events=args.num_events,
        std_eta=args.std_eta,
        n_samples=args.n_samples,
        target_type=args.target_type,
        efp_preset=args.efp_preset,
        efp_beta=args.efp_beta,
        efp_kappa=args.efp_kappa,
        efp_normed=args.efp_normed,
        target_transform=args.target_transform,
    )
    
    # Print results
    print_results(results)
    
    # Plot results
    if not args.no_plot:
        plot_results(results, save_path=args.save_plot)


if __name__ == '__main__':
    main()

