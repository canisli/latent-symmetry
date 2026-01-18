"""Lorentz symmetry loss for kinematic polynomial learning."""

import torch


# ============================================================================
# Lorentz transformation utilities (moved from tagging.utils)
# ============================================================================

def lorentz_squarednorm(v):
    """Lorentz norm, i.e. v^T @ g @ v

    Parameters
    ----------
    v : torch.Tensor
        Tensor of shape (..., 4)

    Returns
    -------
    torch.Tensor
        Lorentz norm of shape (..., )
    """
    t = v[..., 0] * v[..., 0]
    s = (v[..., 1:] * v[..., 1:]).sum(dim=-1)
    return t - s


def lorentz_eye(dims, device=torch.device("cpu"), dtype=torch.float32):
    """
    Create a identity matrix of given shape

    Parameters
    ----------
    dims : tuple
        Dimension of the output tensor, e.g. (2, 3) for a 2x3 matrix
    device : torch.device
        Device to create the tensor on, by default torch.device("cpu")
    dtype : torch.dtype
        Data type of the tensor, by default torch.float32

    Returns
    -------
    torch.Tensor
        Identity matrix of shape (..., 4, 4)
    """
    base_eye = torch.eye(4, dtype=dtype, device=device)
    eye = base_eye.view((1,) * len(dims) + (4, 4)).expand(*dims, 4, 4)
    return eye


def restframe_boost(fourmomenta, checks=False):
    """Construct a Lorentz transformation that boosts four-momenta into their rest frame.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        Tensor of shape (..., 4) representing the four-momenta.
    checks : bool
        If True, perform additional assertion checks on predicted vectors.
        It may cause slowdowns due to GPU/CPU synchronization, use only for debugging.

    Returns
    -------
    trafo : torch.Tensor
        Tensor of shape (..., 4, 4) representing the Lorentz transformation
        that boosts the four-momenta into their rest frame.
    """
    if checks:
        assert (
            lorentz_squarednorm(fourmomenta) > 0
        ).all(), "Trying to boost spacelike vectors into their restframe (not possible). Consider changing the nonlinearity in equivectors."

    # compute relevant quantities
    t0 = fourmomenta.narrow(-1, 0, 1)
    beta = fourmomenta[..., 1:] / t0.clamp_min(1e-10)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    one_minus_beta2 = torch.clamp_min(1 - beta2, min=1e-10)
    gamma = torch.rsqrt(one_minus_beta2)
    boost = -gamma * beta

    # prepare rotation part
    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).expand(
        *fourmomenta.shape[:-1], 3, 3
    )
    scale = (gamma - 1) / torch.clamp_min(beta2, min=1e-10)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    # collect trafo
    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    trafo = torch.cat((row0.unsqueeze(-2), lower), dim=-2)
    return trafo


def rand_wrapper(shape, device, dtype, generator=None):
    """Wrapper for torch.rand to handle generator argument compatibility."""
    if generator is None:
        return torch.rand(shape, device=device, dtype=dtype)
    else:
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def randn_wrapper(shape, device, dtype, generator=None):
    """Wrapper for torch.randn to handle generator argument compatibility."""
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    else:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def sample_rapidity(
    shape: torch.Size,
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Sample rapidity from a clipped gaussian distribution.

    Parameters
    ----------
    shape: torch.Size
        Shape of the output tensor
    std_eta: float
        Standard deviation of the rapidity
    n_max_std_eta: float
        Maximum number of standard deviations for truncation
    device: str
    dtype: torch.dtype
    generator: torch.Generator
    """
    eta = randn_wrapper(shape, device, dtype, generator=generator)
    angle = eta * std_eta
    angle.clamp(min=-std_eta * n_max_std_eta, max=std_eta * n_max_std_eta)
    return angle


def rand_rotation(
    shape: torch.Size,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create rotation matrices embedded in Lorentz transformations.
    The rotations are sampled uniformly using quaternions,
    see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.

    Parameters
    ----------
    shape: torch.Size
        Shape of the transformation matrices
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    # generate random quaternions
    shape2 = torch.Size((*shape, 3))
    u = rand_wrapper(shape2, device, dtype, generator=generator)
    q1 = torch.sqrt(1 - u[..., 0]) * torch.sin(2 * torch.pi * u[..., 1])
    q2 = torch.sqrt(1 - u[..., 0]) * torch.cos(2 * torch.pi * u[..., 1])
    q3 = torch.sqrt(u[..., 0]) * torch.sin(2 * torch.pi * u[..., 2])
    q0 = torch.sqrt(u[..., 0]) * torch.cos(2 * torch.pi * u[..., 2])

    # create rotation matrix from quaternions
    R1 = torch.stack(
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        dim=-1,
    )
    R2 = torch.stack(
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        dim=-1,
    )
    R3 = torch.stack(
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        dim=-1,
    )
    R = torch.stack([R1, R2, R3], dim=-2)

    trafo = torch.eye(4, device=device, dtype=dtype).expand(*shape, 4, 4).clone()
    trafo[..., 1:, 1:] = R
    return trafo


def rand_boost(
    shape: torch.Size,
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Create a general pure boost, i.e. a symmetric Lorentz transformation.

    Parameters
    ----------
    shape: torch.Size
        Shape of the transformation matrices
    std_eta: float
        Standard deviation of rapidity
    n_max_std_eta: float
        Allowed number of standard deviations;
        used to sample from a truncated Gaussian
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    shape = torch.Size((*shape, 3))
    beta = sample_rapidity(
        shape,
        std_eta,
        n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    beta2 = (beta**2).sum(dim=-1, keepdim=True)
    gamma = 1 / (1 - beta2).clamp(min=1e-10).sqrt()
    fourmomenta = torch.cat([gamma, beta], axis=-1)

    boost = restframe_boost(fourmomenta)
    return boost


def rand_lorentz(
    shape: torch.Size,
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Create general Lorentz transformations as rotation * boost.
    Any Lorentz transformation can be expressed in this way,
    see polar decomposition of the Lorentz group.

    Parameters
    ----------
    shape: torch.Size
        Shape of the transformation matrices
    std_eta: float
        Standard deviation of rapidity
    n_max_std_eta: float
        Allowed number of standard deviations;
        used to sample from a truncated Gaussian
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    assert std_eta > 0
    boost = rand_boost(
        shape,
        std_eta,
        n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    rotation = rand_rotation(shape, device, dtype, generator=generator)

    trafo = torch.einsum("...ij,...jk->...ik", rotation, boost)
    return trafo


def lorentz_inverse(L: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a Lorentz transformation matrix.
    
    For a proper Lorentz transformation L, the inverse satisfies:
        L^(-1) = η L^T η
    where η = diag(1, -1, -1, -1) is the Minkowski metric.
    
    Args:
        L: Lorentz transformation matrix of shape (..., 4, 4)
    
    Returns:
        L_inv: Inverse Lorentz transformation of shape (..., 4, 4)
    """
    # Build Minkowski metric η = diag(1, -1, -1, -1)
    eta = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], 
                                   device=L.device, dtype=L.dtype))
    
    # L^(-1) = η L^T η
    # For batched input, we need to handle the transpose correctly
    L_T = L.transpose(-2, -1)
    L_inv = eta @ L_T @ eta
    
    return L_inv


# ============================================================================
# Symmetry loss function
# ============================================================================

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
