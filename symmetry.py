"""Lorentz symmetry loss for kinematic polynomial learning."""

import torch
from tagging.utils.rand_transforms import rand_lorentz


def lorentz_orbit_variance_loss(
    model,
    x: torch.Tensor,
    layer_idx: int,
    std_eta: float = 0.5,
    n_max_std_eta: float = 3.0,
    generator: torch.Generator = None,
    mask: torch.Tensor = None,
):
    """
    Compute symmetry loss by measuring variance of intermediate activations
    under random Lorentz transformations.
    
    The loss encourages the model to produce identical intermediate representations
    for Lorentz-transformed versions of the same input.
    
    Args:
        model: Model with forward_with_intermediate() method
        x: Input 4-vectors of shape (batch_size, num_particles, 4)
        layer_idx: Which layer's activations to compare (-1 for output)
        std_eta: Standard deviation of rapidity for boosts
        n_max_std_eta: Maximum number of standard deviations for truncation
        generator: Optional random generator for reproducibility
        mask: Optional boolean mask of shape (batch_size, num_particles)
              where True indicates real particles
    
    Returns:
        Scalar symmetry loss: 0.5 * mean(||h1 - h2||^2)
    """
    B, N, D = x.shape
    assert D == 4, f"Expected 4-vectors, got dimension {D}"
    device, dtype = x.device, x.dtype
    
    # Sample two random Lorentz transformations: (B, 4, 4)
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
    # x: (B, N, 4) -> (B, N, 4, 1)
    # L: (B, 4, 4) -> (B, 1, 4, 4) for broadcasting
    # Result: (B, N, 4)
    x_rot1 = torch.matmul(L1.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.matmul(L2.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)
    
    # Get intermediate activations
    h1 = model.forward_with_intermediate(x_rot1, layer_idx, mask=mask)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx, mask=mask)
    
    # Handle per-particle representations (pre-pooling layers)
    # These have shape (B, N, hidden) - compare particle-by-particle without pooling
    if h1.dim() == 3:
        if mask is None:
            mask = torch.any(x != 0.0, dim=-1)  # (B, N)
        
        # Compute per-particle squared differences: (B, N, hidden) -> (B, N)
        diff = h1 - h2
        per_particle_loss = diff.pow(2).sum(dim=-1)  # (B, N)
        
        # Average over valid particles across all events
        mask_float = mask.float()  # (B, N)
        total_valid = mask_float.sum().clamp(min=1.0)
        loss = 0.5 * (per_particle_loss * mask_float).sum() / total_valid
        
        return loss
    
    # Post-pooling layers: h1 and h2 have shape (B, hidden)
    # Compute variance loss: 0.5 * ||h1 - h2||^2
    diff = h1 - h2
    loss = 0.5 * (diff.pow(2).sum(dim=-1)).mean()
    
    return loss
