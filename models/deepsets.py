"""DeepSets model for permutation-invariant set learning.

DeepSets avoids the over-smoothing problem of transformers by processing
particles independently before pooling. This preserves individual particle
information needed for tasks like kinematic polynomial prediction.
"""

import torch
import torch.nn as nn
from typing import Optional


class DeepSets(nn.Module):
    """DeepSets architecture: phi(particles) -> pool -> rho(pool) -> output_proj.
    
    Processes each particle independently with phi network, pools via sum/mean,
    then applies rho network followed by output projection.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels per particle (e.g., 4 for E,px,py,pz)
    out_channels : int
        Number of output channels
    hidden_channels : int
        Hidden dimension for phi and rho networks
    num_phi_layers : int
        Number of hidden layers in the per-particle phi network
    num_rho_layers : int
        Number of hidden layers in the post-pooling rho network
    pool_mode : str
        Pooling mode: 'sum' or 'mean'
    
    Layer Indexing for forward_with_intermediate:
        For num_phi_layers=4, num_rho_layers=4:
        - layer_idx 1-4: phi hidden layer activations (per-particle)
        - layer_idx 5: post-pooling (pre-rho)
        - layer_idx 6-9: rho hidden layer activations
        - layer_idx -1: final output (after output_proj)
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
        self.num_phi_layers = num_phi_layers
        self.num_rho_layers = num_rho_layers
        self.hidden_channels = hidden_channels
        
        # Build phi network (per-particle encoder) - store layers individually
        self.phi_layers = nn.ModuleList()
        self.phi_layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_phi_layers - 1):
            self.phi_layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.phi_act = nn.ReLU()
        
        # Build rho network (post-pooling hidden layers) - store layers individually
        self.rho_layers = nn.ModuleList()
        for _ in range(num_rho_layers):
            self.rho_layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.rho_act = nn.ReLU()
        
        # Separate output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
    
    def _apply_mask_and_pool(
        self, 
        h: torch.Tensor, 
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply masking and pooling to per-particle representations."""
        if mask is None:
            mask = torch.any(inputs != 0.0, dim=-1)  # (..., num_particles)
        
        mask_float = mask.float().unsqueeze(-1)  # (..., num_particles, 1)
        h = h * mask_float
        
        if self.pool_mode == 'sum':
            h = h.sum(dim=-2)  # (..., hidden)
        elif self.pool_mode == 'mean':
            valid_counts = mask_float.sum(dim=-2).clamp(min=1.0)
            h = h.sum(dim=-2) / valid_counts
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")
        
        return h
    
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
        # Per-particle encoding through phi layers
        h = inputs
        for layer in self.phi_layers:
            h = self.phi_act(layer(h))
        
        # Pool over particles
        h = self._apply_mask_and_pool(h, inputs, mask)
        
        # Post-pooling processing through rho layers
        for layer in self.rho_layers:
            h = self.rho_act(layer(h))
        
        # Output projection
        return self.output_proj(h)
    
    def forward_with_intermediate(
        self,
        inputs: torch.Tensor,
        layer_idx: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning intermediate activations at specified layer.
        
        Parameters
        ----------
        inputs : Tensor
            Input data with shape (..., num_particles, in_channels)
        layer_idx : int
            Layer index to return activations from:
            - 1..num_phi_layers: phi layer activations (per-particle)
            - num_phi_layers + 1: post-pooling (pre-rho)
            - num_phi_layers + 2..num_phi_layers + num_rho_layers + 1: rho activations
            - -1: final output
        mask : Optional[Tensor]
            Boolean mask with shape (..., num_particles)
        
        Returns
        -------
        activations : Tensor
            Intermediate activations at the specified layer
        """
        if layer_idx == -1:
            return self.forward(inputs, mask)
        
        pool_layer_idx = self.num_phi_layers + 1
        max_rho_layer_idx = self.num_phi_layers + self.num_rho_layers + 1
        
        if layer_idx < 1 or layer_idx > max_rho_layer_idx:
            raise ValueError(
                f"layer_idx must be between 1 and {max_rho_layer_idx}, or -1. "
                f"Got {layer_idx}"
            )
        
        # Process through phi layers
        h = inputs
        for i, layer in enumerate(self.phi_layers, start=1):
            h = self.phi_act(layer(h))
            if i == layer_idx:
                return h  # Return per-particle activations
        
        # Pool over particles
        h = self._apply_mask_and_pool(h, inputs, mask)
        
        if layer_idx == pool_layer_idx:
            return h  # Return post-pooling activations
        
        # Process through rho layers
        rho_start_idx = pool_layer_idx + 1
        for i, layer in enumerate(self.rho_layers):
            h = self.rho_act(layer(h))
            if rho_start_idx + i == layer_idx:
                return h  # Return rho layer activations
        
        # Should not reach here
        raise RuntimeError(f"Failed to return activations for layer_idx={layer_idx}")
