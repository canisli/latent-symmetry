"""
SO(3) symmetry loss functions for set inputs (e.g., point clouds).

These functions measure how invariant model activations are to SO(3) rotations
applied uniformly to all points in a set.

Key difference from symmetry.py:
- For layers BEFORE pooling (phi layers), compute per-point loss and average
- For layers AFTER pooling (rho layers), compute standard loss on aggregated embeddings
"""

import torch
from symmetry import sample_so3_rotation


def so3_relative_symmetry_loss_sets(
    model, 
    points, 
    layer_idx, 
    generator=None, 
    eps=1e-8
):
    """
    Compute relative symmetry loss for set inputs at a given layer.
    
    For layers before pooling (phi network):
        - Embeddings have shape (batch, n_points, hidden_dim)
        - Compute per-point relative loss, then average across points
    
    For layers after pooling (rho network):
        - Embeddings have shape (batch, hidden_dim)
        - Compute standard relative loss
    
    Args:
        model: DeepSets model with forward_with_intermediate and is_layer_before_pooling methods
        points: Input tensor of shape (batch_size, n_points, 3)
        layer_idx: Layer index (1-based for layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
    
    Returns:
        Scalar tensor with mean relative symmetry loss
    """
    B, N, D = points.shape
    assert D == 3, f"Expected 3D points, got {D}D"
    device, dtype = points.device, points.dtype

    # Sample two random rotations (one per batch element)
    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    # Apply same rotation to all points in each sample
    # R: (B, 3, 3), points: (B, N, 3) -> points.transpose(-1, -2): (B, 3, N)
    # R @ points^T: (B, 3, N) -> transpose back: (B, N, 3)
    points_rot1 = torch.bmm(R1, points.transpose(-1, -2)).transpose(-1, -2)
    points_rot2 = torch.bmm(R2, points.transpose(-1, -2)).transpose(-1, -2)

    # Get embeddings at specified layer
    h1 = model.forward_with_intermediate(points_rot1, layer_idx)
    h2 = model.forward_with_intermediate(points_rot2, layer_idx)

    # Check if layer is before pooling (per-point embeddings)
    before_pooling = model.is_layer_before_pooling(layer_idx) if layer_idx != -1 else False
    
    if before_pooling:
        # h1, h2: (batch, n_points, hidden_dim)
        # Compute per-point relative loss
        diff_sq = (h1 - h2).pow(2).sum(dim=-1)  # (batch, n_points)
        norm_sq_sum = h1.pow(2).sum(dim=-1) + h2.pow(2).sum(dim=-1)  # (batch, n_points)
        relative_loss = diff_sq / (norm_sq_sum + eps)  # (batch, n_points)
        # Average across both batch and points
        return relative_loss.mean()
    else:
        # h1, h2: (batch, hidden_dim)
        # Standard relative loss
        diff_sq = (h1 - h2).pow(2).sum(dim=-1)  # (batch,)
        norm_sq_sum = h1.pow(2).sum(dim=-1) + h2.pow(2).sum(dim=-1)  # (batch,)
        relative_loss = diff_sq / (norm_sq_sum + eps)  # (batch,)
        return relative_loss.mean()


def so3_orbit_variance_loss_sets(model, points, layer_idx, generator=None):
    """
    Symmetry loss using absolute difference (not normalized).
    
    For layers before pooling (phi network):
        - Compute per-point ||h1 - h2||^2, average across points
    
    For layers after pooling (rho network):
        - Compute standard ||h1 - h2||^2
    
    Args:
        model: DeepSets model with forward_with_intermediate and is_layer_before_pooling methods
        points: Input tensor of shape (batch_size, n_points, 3)
        layer_idx: Layer index (1-based for layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
    
    Returns:
        Scalar tensor with mean orbit variance loss
    """
    B, N, D = points.shape
    assert D == 3
    device, dtype = points.device, points.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    points_rot1 = torch.bmm(R1, points.transpose(-1, -2)).transpose(-1, -2)
    points_rot2 = torch.bmm(R2, points.transpose(-1, -2)).transpose(-1, -2)

    h1 = model.forward_with_intermediate(points_rot1, layer_idx)
    h2 = model.forward_with_intermediate(points_rot2, layer_idx)

    before_pooling = model.is_layer_before_pooling(layer_idx) if layer_idx != -1 else False
    
    if before_pooling:
        # h1, h2: (batch, n_points, hidden_dim)
        diff = h1 - h2
        # Sum over hidden_dim, mean over points and batch
        return 0.5 * (diff.pow(2).sum(dim=-1)).mean()
    else:
        # h1, h2: (batch, hidden_dim)
        diff = h1 - h2
        return 0.5 * (diff.pow(2).sum(dim=-1)).mean()


def so3_relative_symmetry_loss_mlp_sets(
    model, 
    points_flat, 
    n_points,
    layer_idx, 
    generator=None, 
    eps=1e-8
):
    """
    Compute relative symmetry loss for MLP with flattened set inputs.
    
    The MLP takes flattened points (batch, n_points * 3) but we need to
    apply the same rotation to all points in each sample.
    
    Args:
        model: MLP model with forward_with_intermediate method
        points_flat: Input tensor of shape (batch_size, n_points * 3)
        n_points: Number of points per sample
        layer_idx: Layer index (1-based for layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
    
    Returns:
        Scalar tensor with mean relative symmetry loss
    """
    B = points_flat.shape[0]
    device, dtype = points_flat.device, points_flat.dtype
    
    # Reshape to (batch, n_points, 3)
    points = points_flat.reshape(B, n_points, 3)
    
    # Sample rotations
    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    # Apply rotations
    points_rot1 = torch.bmm(R1, points.transpose(-1, -2)).transpose(-1, -2)
    points_rot2 = torch.bmm(R2, points.transpose(-1, -2)).transpose(-1, -2)

    # Flatten back for MLP
    points_rot1_flat = points_rot1.reshape(B, -1)
    points_rot2_flat = points_rot2.reshape(B, -1)

    # Get embeddings
    h1 = model.forward_with_intermediate(points_rot1_flat, layer_idx)
    h2 = model.forward_with_intermediate(points_rot2_flat, layer_idx)

    # Standard relative loss (MLP has no per-point structure)
    diff_sq = (h1 - h2).pow(2).sum(dim=-1)
    norm_sq_sum = h1.pow(2).sum(dim=-1) + h2.pow(2).sum(dim=-1)
    relative_loss = diff_sq / (norm_sq_sum + eps)
    
    return relative_loss.mean()


if __name__ == '__main__':
    import torch
    from models import DeepSets, MLP
    
    print("Testing symmetry loss functions for sets...")
    
    # Create DeepSets model
    model = DeepSets(phi_dims=[3, 64, 64], rho_dims=[64, 32, 1], pooling='sum')
    
    # Create test input
    points = torch.randn(8, 50, 3)  # batch=8, n_points=50, dim=3
    
    print("\nDeepSets symmetry loss per layer:")
    for layer_idx in range(1, model.total_layers + 1):
        loss = so3_relative_symmetry_loss_sets(model, points, layer_idx)
        before_pool = model.is_layer_before_pooling(layer_idx)
        print(f"  Layer {layer_idx}: {loss.item():.4f} (before pooling: {before_pool})")
    
    loss = so3_relative_symmetry_loss_sets(model, points, -1)
    print(f"  Layer -1: {loss.item():.4f}")
    
    # Test MLP version
    print("\nMLP symmetry loss per layer:")
    mlp = MLP([150, 128, 64, 1])  # 50*3 = 150 input dim
    points_flat = points.reshape(8, -1)
    
    for layer_idx in range(1, mlp.num_linear_layers + 1):
        loss = so3_relative_symmetry_loss_mlp_sets(mlp, points_flat, n_points=50, layer_idx=layer_idx)
        print(f"  Layer {layer_idx}: {loss.item():.4f}")
    
    loss = so3_relative_symmetry_loss_mlp_sets(mlp, points_flat, n_points=50, layer_idx=-1)
    print(f"  Layer -1: {loss.item():.4f}")
    
    print("\nDone!")

