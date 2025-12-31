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


def so3_orbit_variance_loss_two_layers(model, x, layer_idx_1, layer_idx_2, generator=None):
    """
    Compute symmetry loss for two layers simultaneously.
    
    Args:
        model: The MLP model
        x: Input tensor of shape (batch_size, 3)
        layer_idx_1: Index of first layer (last hidden layer)
        layer_idx_2: Index of second layer (output layer, typically -1)
        generator: Random generator for reproducibility
    
    Returns:
        Tuple of (loss_1, loss_2) - symmetry losses for each layer separately
    """
    B, D = x.shape
    assert D == 3
    device, dtype = x.device, x.dtype

    R1 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)
    R2 = sample_so3_rotation(B, device=device, dtype=dtype, generator=generator)

    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)

    # Compute activations for first layer
    h1_layer1 = model.forward_with_intermediate(x_rot1, layer_idx_1)
    h2_layer1 = model.forward_with_intermediate(x_rot2, layer_idx_1)
    diff1 = h1_layer1 - h2_layer1
    loss_1 = 0.5 * (diff1.pow(2).sum(dim=1)).mean()

    # Compute activations for second layer
    h1_layer2 = model.forward_with_intermediate(x_rot1, layer_idx_2)
    h2_layer2 = model.forward_with_intermediate(x_rot2, layer_idx_2)
    diff2 = h1_layer2 - h2_layer2
    loss_2 = 0.5 * (diff2.pow(2).sum(dim=1)).mean()

    return loss_1, loss_2

