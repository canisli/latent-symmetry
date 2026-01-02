"""DeepSets model for permutation-invariant set learning.

DeepSets avoids the over-smoothing problem of transformers by processing
particles independently before pooling. This preserves individual particle
information needed for tasks like kinematic polynomial prediction.
"""

import torch
import torch.nn as nn
from typing import Optional


class DeepSets(nn.Module):
    """DeepSets architecture: phi(particles) -> sum -> rho(sum).
    
    Processes each particle independently with phi network, pools via sum,
    then applies rho network to produce output.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels per particle (e.g., 4 for E,px,py,pz)
    out_channels : int
        Number of output channels
    hidden_channels : int
        Hidden dimension for phi and rho networks
    num_phi_layers : int
        Number of layers in the per-particle phi network
    num_rho_layers : int
        Number of layers in the post-pooling rho network
    pool_mode : str
        Pooling mode: 'sum' or 'mean'
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_phi_layers: int = 3,
        num_rho_layers: int = 3,
        pool_mode: str = 'sum',
    ):
        super().__init__()
        self.pool_mode = pool_mode
        
        # Build phi network (per-particle encoder)
        phi_layers = []
        phi_layers.append(nn.Linear(in_channels, hidden_channels))
        phi_layers.append(nn.ReLU())
        for _ in range(num_phi_layers - 1):
            phi_layers.append(nn.Linear(hidden_channels, hidden_channels))
            phi_layers.append(nn.ReLU())
        self.phi = nn.Sequential(*phi_layers)
        
        # Build rho network (post-pooling)
        rho_layers = []
        for _ in range(num_rho_layers - 1):
            rho_layers.append(nn.Linear(hidden_channels, hidden_channels))
            rho_layers.append(nn.ReLU())
        rho_layers.append(nn.Linear(hidden_channels, out_channels))
        self.rho = nn.Sequential(*rho_layers)
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        inputs : Tensor
            Input data with shape (..., num_particles, in_channels)
        mask : Optional[Tensor]
            Boolean mask with shape (..., num_particles) where True indicates
            real particles. If None, auto-detects from zero inputs.
        
        Returns
        -------
        outputs : Tensor
            Output with shape (..., out_channels)
        """
        # Per-particle encoding
        h = self.phi(inputs)  # (..., num_particles, hidden)
        
        # Handle masking for padded particles
        if mask is None:
            mask = torch.any(inputs != 0.0, dim=-1)  # (..., num_particles)
        
        # Apply mask
        mask_float = mask.float().unsqueeze(-1)  # (..., num_particles, 1)
        h = h * mask_float
        
        # Pool over particles
        if self.pool_mode == 'sum':
            h = h.sum(dim=-2)  # (..., hidden)
        elif self.pool_mode == 'mean':
            valid_counts = mask_float.sum(dim=-2).clamp(min=1.0)
            h = h.sum(dim=-2) / valid_counts
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")
        
        # Post-pooling processing
        return self.rho(h)

