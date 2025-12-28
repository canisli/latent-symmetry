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


def so3_orbit_variance_loss(model, x, layer_idx, aux_head=None, generator=None):
    """
    Compute SO(3) orbit variance loss at a specified layer.
    
    Args:
        model: MLP model with forward_with_intermediate method
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index for intermediate activations
        aux_head: Optional nn.Linear auxiliary head to apply to intermediate
                  activations before computing variance. If None, uses raw
                  intermediate activations.
        generator: Optional torch.Generator for reproducible rotations
    
    Returns:
        Scalar loss tensor
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

    # Apply auxiliary head if provided
    if aux_head is not None:
        h1 = aux_head(h1)
        h2 = aux_head(h2)

    diff = h1 - h2
    return 0.5 * (diff.pow(2).sum(dim=1)).mean()

