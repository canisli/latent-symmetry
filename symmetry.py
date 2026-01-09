import torch


def sample_so3_rotation(batch_size, device="cpu", dtype=torch.float32, eps=1e-8, generator=None):
    # Haar on SO(3) via uniform unit quaternions (S^3); q and -q map to same rotation.
    if generator is None:
        q = torch.randn(batch_size, 4, device=device, dtype=dtype)
    else:
        q = torch.randn(batch_size, 4, device=device, dtype=dtype, generator=generator)
    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))

    w, x, y, z = q.unbind(dim=-1)

    R = torch.empty(batch_size, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def so3_orbit_variance_loss(model, x, layer_idx, generator=None):
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    h1 = model.forward_with_intermediate(x_rot1, layer_idx)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)

    diff = h1 - h2
    return 0.5 * (diff.pow(2).sum(dim=1)).mean()


def so3_relative_symmetry_loss(model, x, layer_idx, generator=None, eps=1e-8):
    """
    Compute relative symmetry loss at a given layer.
    
    Unlike so3_orbit_variance_loss which computes ||h1 - h2||^2,
    this computes a scale-invariant relative loss:
        ||h1 - h2||^2 / (||h1||^2 + ||h2||^2 + eps)
    
    This makes the loss comparable across layers with different activation magnitudes.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index (1-based for hidden layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
    
    Returns:
        Scalar tensor with mean relative symmetry loss
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    h1 = model.forward_with_intermediate(x_rot1, layer_idx)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)

    # Compute ||h1 - h2||^2 per sample
    diff_sq = (h1 - h2).pow(2).sum(dim=1)
    
    # Compute ||h1||^2 + ||h2||^2 per sample
    norm_sq_sum = h1.pow(2).sum(dim=1) + h2.pow(2).sum(dim=1)
    
    # Relative loss: ||h1 - h2||^2 / (||h1||^2 + ||h2||^2 + eps)
    #
    # POTENTIAL ISSUE: When model outputs are very small (close to zero), this
    # formula can give misleadingly large values. For example:
    #   - If h1 ≈ 0.001 and h2 ≈ -0.001 (opposite signs, small magnitude):
    #     diff_sq ≈ (0.002)^2 = 4e-6
    #     norm_sq_sum ≈ 1e-6 + 1e-6 = 2e-6
    #     relative_loss ≈ 4e-6 / 2e-6 = 2.0
    #   - This suggests non-invariance, but the absolute difference is tiny!
    #
    # For relative_loss ≈ 1.0, we need: ||h1 - h2||^2 ≈ ||h1||^2 + ||h2||^2
    # For scalar outputs: (h1 - h2)^2 ≈ h1^2 + h2^2, which means h1 * h2 ≈ 0
    # This happens when h1 and h2 have opposite signs or one is close to zero.
    #
    # If the model outputs small values with opposite signs for rotated inputs,
    # this indicates the model hasn't learned perfect invariance, even though
    # the field is invariant. This could happen if:
    #   1. The model is trained without symmetry penalty (baseline)
    #   2. The outputs are near zero and numerical precision issues occur
    #   3. The normalization is misleading when outputs are very small
    #
    # To diagnose, check if outputs are small and have opposite signs.
    # Consider using a different normalization or adding a minimum threshold
    # to the denominator when outputs are very small.
    relative_loss = diff_sq / (norm_sq_sum + eps)
    
    return relative_loss.mean()


def so3_maxnorm_relative_loss(model, x, layer_idx, generator=None, eps=1e-8):
    """
    Alternative relative symmetry loss using max norm normalization.
    
    Formula: ||h1 - h2||^2 / (max(||h1||^2, ||h2||^2) + eps)
    
    This is more robust to small outputs than the sum-based normalization.
    When outputs are small, using max instead of sum prevents the denominator
    from being too small, making the loss more stable.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index (1-based for hidden layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
    
    Returns:
        Scalar tensor with mean relative symmetry loss
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    h1 = model.forward_with_intermediate(x_rot1, layer_idx)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)

    diff_sq = (h1 - h2).pow(2).sum(dim=1)
    norm_sq_1 = h1.pow(2).sum(dim=1)
    norm_sq_2 = h2.pow(2).sum(dim=1)
    norm_sq_max = torch.maximum(norm_sq_1, norm_sq_2)
    
    relative_loss = diff_sq / (norm_sq_max + eps)
    return relative_loss.mean()


def so3_cosine_symmetry_loss(model, x, layer_idx, generator=None, eps=1e-8):
    """
    Symmetry loss based on cosine similarity.
    
    Formula: 1 - cosine_similarity(h1, h2)
    where cosine_similarity = (h1 · h2) / (||h1|| * ||h2|| + eps)
    
    This measures the angle between h1 and h2, independent of magnitude.
    Perfect invariance gives cosine_similarity = 1, so loss = 0.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index (1-based for hidden layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
    
    Returns:
        Scalar tensor with mean cosine-based symmetry loss
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    h1 = model.forward_with_intermediate(x_rot1, layer_idx)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)

    # Compute cosine similarity per sample
    dot_product = (h1 * h2).sum(dim=1)
    norm1 = h1.norm(dim=1)
    norm2 = h2.norm(dim=1)
    cosine_sim = dot_product / (norm1 * norm2 + eps)
    
    # Loss is 1 - cosine similarity (0 for perfect alignment, 1 for opposite, 2 for orthogonal)
    loss = 1 - cosine_sim
    return loss.mean()


def so3_mean_magnitude_relative_loss(model, x, layer_idx, generator=None, eps=1e-8):
    """
    Relative symmetry loss normalized by mean magnitude.
    
    Formula: ||h1 - h2|| / (mean(||h1||, ||h2||) + eps)
    
    This uses the mean of the magnitudes as the normalization factor,
    which is more stable than sum-based normalization for small values.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index (1-based for hidden layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
    
    Returns:
        Scalar tensor with mean relative symmetry loss
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    h1 = model.forward_with_intermediate(x_rot1, layer_idx)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)

    diff_norm = (h1 - h2).norm(dim=1)
    norm1 = h1.norm(dim=1)
    norm2 = h2.norm(dim=1)
    mean_norm = (norm1 + norm2) / 2
    
    relative_loss = diff_norm / (mean_norm + eps)
    return relative_loss.mean()


def so3_adaptive_threshold_loss(model, x, layer_idx, generator=None, eps=1e-8, threshold_factor=1e-3):
    """
    Relative symmetry loss with adaptive threshold based on batch statistics.
    
    Formula: ||h1 - h2||^2 / (||h1||^2 + ||h2||^2 + threshold)
    where threshold = max(eps, threshold_factor * mean(||h1||^2 + ||h2||^2))
    
    This adds a minimum threshold to the denominator based on the batch mean,
    preventing division by very small values while maintaining scale-invariance.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index (1-based for hidden layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
        threshold_factor: Factor to multiply batch mean for threshold (default: 1e-3)
    
    Returns:
        Scalar tensor with mean relative symmetry loss
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    h1 = model.forward_with_intermediate(x_rot1, layer_idx)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)

    diff_sq = (h1 - h2).pow(2).sum(dim=1)
    norm_sq_sum = h1.pow(2).sum(dim=1) + h2.pow(2).sum(dim=1)
    
    # Adaptive threshold based on batch mean
    mean_norm_sq = norm_sq_sum.mean()
    threshold = max(eps, threshold_factor * mean_norm_sq)
    
    relative_loss = diff_sq / (norm_sq_sum + threshold)
    return relative_loss.mean()


def so3_orbit_variance_loss_multi(model, x, layer_idx, num_rotations=5, generator=None):
    """
    Symmetry loss computed as variance across multiple rotations (orbit).
    
    Formula: Var({h_i}) where h_i = model(R_i @ x) for i=1..num_rotations
    
    This measures the variance of outputs across multiple random rotations,
    giving a more stable estimate of invariance than pairwise comparisons.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index (1-based for hidden layers, -1 for output)
        num_rotations: Number of random rotations to sample (default: 5)
        generator: Optional torch.Generator for reproducible rotations
    
    Returns:
        Scalar tensor with mean variance across orbit
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    # Sample multiple rotations
    rotations = []
    for _ in range(num_rotations):
        R = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
        rotations.append(R)
    
    # Compute outputs for each rotation
    outputs = []
    for R in rotations:
        x_rot = torch.bmm(R, x.unsqueeze(-1)).squeeze(-1)
        h = model.forward_with_intermediate(x_rot, layer_idx)
        outputs.append(h)
    
    # Stack outputs: shape (num_rotations, batch_size, dim)
    outputs = torch.stack(outputs, dim=0)
    
    # Compute variance across rotations per sample
    # Variance = mean((h_i - mean(h))^2)
    mean_output = outputs.mean(dim=0)  # (batch_size, dim)
    variance = ((outputs - mean_output.unsqueeze(0)) ** 2).mean(dim=0).sum(dim=1)  # (batch_size,)
    
    return variance.mean()


def so3_coefficient_of_variation_loss(model, x, layer_idx, generator=None, eps=1e-8):
    """
    Symmetry loss based on coefficient of variation.
    
    Formula: std({h1, h2}) / (mean(|h1|, |h2|) + eps)
    
    This measures the relative variability between h1 and h2,
    normalized by their mean magnitude. Similar to coefficient of variation
    in statistics.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index (1-based for hidden layers, -1 for output)
        generator: Optional torch.Generator for reproducible rotations
        eps: Small constant to avoid division by zero
    
    Returns:
        Scalar tensor with mean coefficient of variation loss
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    h1 = model.forward_with_intermediate(x_rot1, layer_idx)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)

    # For each sample, compute std and mean across the two outputs
    # Stack h1 and h2: (2, batch_size, dim)
    outputs = torch.stack([h1, h2], dim=0)
    
    # Mean and std across the rotation dimension (dim=0)
    mean_output = outputs.mean(dim=0)  # (batch_size, dim)
    std_output = outputs.std(dim=0)  # (batch_size, dim)
    
    # Coefficient of variation per sample
    mean_magnitude = mean_output.norm(dim=1)  # (batch_size,)
    std_magnitude = std_output.norm(dim=1)  # (batch_size,)
    
    cv = std_magnitude / (mean_magnitude + eps)
    return cv.mean()

