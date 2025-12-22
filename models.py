import torch
import torch.nn as nn


def sample_so3_rotation(batch_size, device="cpu", dtype=torch.float32, eps=1e-8):
    # Haar on SO(3) via uniform unit quaternions (S^3); q and -q map to same rotation.
    q = torch.randn(batch_size, 4, device=device, dtype=dtype)
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


def so3_orbit_variance_loss(model, x, layer_idx):
    """
    Compute SO(3) orbit variance loss: 1/2 ||h(R1*x) - h(R2*x)||^2
    
    For each sample in the batch, samples two rotations R1 and R2 from the Haar measure,
    applies them to the input, and computes the variance of activations at the specified layer.
    
    Args:
        model: MLP model instance
        x: Input tensor of shape (batch_size, 3)
        layer_idx: Layer index to compute loss at (see MLP.forward_with_intermediate)
    
    Returns:
        Scalar loss value (mean over batch)
    """
    batch_size = x.shape[0]
    device = x.device
    
    # Sample rotations R1 and R2 for each sample in the batch
    R1 = sample_so3_rotation(batch_size, device=device)
    R2 = sample_so3_rotation(batch_size, device=device)
    
    # Apply rotations: x_rot = (R @ x^T)^T = x @ R^T
    # x has shape (batch_size, 3), R has shape (batch_size, 3, 3)
    # We want to compute R @ x for each sample
    x_rot1 = torch.bmm(R1, x.unsqueeze(-1)).squeeze(-1)  # (batch_size, 3)
    x_rot2 = torch.bmm(R2, x.unsqueeze(-1)).squeeze(-1)  # (batch_size, 3)
    
    # Get activations at specified layer
    h1 = model.forward_with_intermediate(x_rot1, layer_idx)  # (batch_size, hidden_dim)
    h2 = model.forward_with_intermediate(x_rot2, layer_idx)  # (batch_size, hidden_dim)
    
    # Compute 1/2 ||h(R1*x) - h(R2*x)||^2 for each sample
    diff = h1 - h2  # (batch_size, hidden_dim)
    loss_per_sample = 0.5 * torch.sum(diff ** 2, dim=1)  # (batch_size,)
    
    # Return mean over batch
    return loss_per_sample.mean()
    

class MLP(nn.Module):
    # check weight inititialization

    def __init__(self, d_in, hidden_dims: list[int], d_out):
        super().__init__()
        
        self.d_in = d_in
        self.hidden_dims = hidden_dims
        self.d_out = d_out
        
        self.embed = nn.Linear(d_in, hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.hidden_layers.append(nn.ReLU())
        
        self.proj = nn.Linear(hidden_dims[-1], d_out)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.hidden_layers:
            h = layer(h)
        return self.proj(h)
    
    def forward_with_intermediate(self, x, layer_idx):
        """
        Forward pass that returns intermediate activations at specified layer.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            layer_idx: Layer index (0-based):
                - i=0: after embed layer
                - i=1, 2, ...: after the 1st, 2nd, ... hidden layers
                - i=len(hidden_layers)//2 + 1 or i=-1: after all hidden layers (before proj)
        
        Returns:
            Activation tensor at the specified layer
        """
        # Convert -1 to final layer index
        num_hidden_layers = len(self.hidden_layers) // 2
        if layer_idx == -1:
            layer_idx = num_hidden_layers + 1
        
        # Layer 0: after embed
        h = self.embed(x)
        if layer_idx == 0:
            return h
        
        # Process hidden layers
        # hidden_layers contains pairs: [Linear, ReLU, Linear, ReLU, ...]
        # After each ReLU (odd indices), we complete a hidden layer
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            # After ReLU at odd index, we're at layer (i//2 + 1)
            # i=1 -> layer 1, i=3 -> layer 2, etc.
            if i % 2 == 1:  # After ReLU
                current_layer = i // 2 + 1
                if current_layer == layer_idx:
                    return h
        
        # If we get here, we've processed all hidden layers
        # layer_idx == num_hidden_layers + 1 is the final layer (after all hidden layers)
        if layer_idx == num_hidden_layers + 1:
            return h
        
        raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be between 0 and {num_hidden_layers + 1}, or -1")

