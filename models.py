import torch
import torch.nn as nn
from typing import Literal


class MLP(nn.Module):
    def __init__(self, dims: list[int], act=nn.ReLU):
        """
        Args:
            dims: List of layer sizes, including input and output.
                  Example: [3, 128, 128, 1]
            act: Activation class to use between linear layers (default: ReLU).
        """
        super().__init__()
        
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
            if i < len(dims) - 1:
                layers.append(act())

        self.layers = nn.Sequential(*layers)
        self.num_linear_layers = len(dims) - 1

    def forward(self, x):
        return self.layers(x)
    
    def forward_with_intermediate(self, x, layer_idx):
        """
        Forward pass that returns intermediate activations at specified layer.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            layer_idx: Layer index (1-based):
                - i=1: after activation of first linear layer
                - i=2: after activation of second linear layer
                - ...
                - i=-1: final output of the network
        
        Returns:
            Activation tensor at the specified layer
        """
        if layer_idx == -1:
            return self.forward(x)

        if layer_idx < 1 or layer_idx > self.num_linear_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be between 1 and {self.num_linear_layers}, or -1")

        h = x
        current_linear = 0
        layers = list(self.layers)

        for i, layer in enumerate(layers):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                current_linear += 1
                if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                    h = layers[i + 1](h)
            
                if current_linear == layer_idx:
                    return h

        raise RuntimeError("Failed to retrieve intermediate activation.")


class DeepSets(nn.Module):
    """
    DeepSets architecture for permutation-invariant set functions.
    
    Architecture:
        phi: per-point encoder (shared MLP applied to each point)
        pooling: sum or mean aggregation
        rho: set-level decoder MLP
    
    Layer indexing for forward_with_intermediate:
        - Layers 1..num_phi_layers: phi layers (per-point, shape (batch, n_points, H))
        - Layer num_phi_layers + 1: after pooling (shape (batch, H))
        - Layers num_phi_layers + 2..: rho layers (shape (batch, H'))
        - Layer -1: final output
    """
    
    def __init__(
        self,
        phi_dims: list[int],
        rho_dims: list[int],
        pooling: Literal['sum', 'mean'] = 'sum',
        act=nn.ReLU
    ):
        """
        Args:
            phi_dims: Dimensions for phi network including input (e.g., [3, 64, 64])
            rho_dims: Dimensions for rho network including output (e.g., [64, 64, 1])
                      Note: first dim of rho_dims should match last dim of phi_dims
            pooling: 'sum' or 'mean' aggregation
            act: Activation class to use between linear layers
        """
        super().__init__()
        
        # Build phi network (per-point encoder)
        phi_layers = []
        for i in range(1, len(phi_dims)):
            phi_layers.append(nn.Linear(phi_dims[i-1], phi_dims[i]))
            if i < len(phi_dims) - 1:
                phi_layers.append(act())
        self.phi = nn.Sequential(*phi_layers)
        self.num_phi_linear_layers = len(phi_dims) - 1
        
        # Build rho network (set-level decoder)
        rho_layers = []
        for i in range(1, len(rho_dims)):
            rho_layers.append(nn.Linear(rho_dims[i-1], rho_dims[i]))
            if i < len(rho_dims) - 1:
                rho_layers.append(act())
        self.rho = nn.Sequential(*rho_layers)
        self.num_rho_linear_layers = len(rho_dims) - 1
        
        self.pooling = pooling
        self.num_linear_layers = self.num_phi_linear_layers + self.num_rho_linear_layers
        
        # Total layers for intermediate access:
        # phi layers + 1 (pooling) + rho layers
        self.total_layers = self.num_phi_linear_layers + 1 + self.num_rho_linear_layers
    
    def _apply_phi_with_intermediate(self, x, target_layer):
        """
        Apply phi network and optionally return intermediate activations.
        
        Args:
            x: Input tensor of shape (batch, n_points, d_in)
            target_layer: Layer index (1-based), or None to run full phi
        
        Returns:
            Output tensor and whether target was reached
        """
        h = x
        current_linear = 0
        layers = list(self.phi)
        
        for i, layer in enumerate(layers):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                current_linear += 1
                # Apply activation if next layer exists and is activation
                if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                    h = layers[i + 1](h)
                
                if target_layer is not None and current_linear == target_layer:
                    return h, True
        
        return h, False
    
    def _apply_rho_with_intermediate(self, x, target_layer):
        """
        Apply rho network and optionally return intermediate activations.
        
        Args:
            x: Input tensor of shape (batch, d_in)
            target_layer: Layer index (1-based within rho), or None to run full rho
        
        Returns:
            Output tensor and whether target was reached
        """
        h = x
        current_linear = 0
        layers = list(self.rho)
        
        for i, layer in enumerate(layers):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                current_linear += 1
                # Apply activation if next layer exists and is activation
                if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                    h = layers[i + 1](h)
                
                if target_layer is not None and current_linear == target_layer:
                    return h, True
        
        return h, False
    
    def _pool(self, h):
        """Apply pooling aggregation."""
        if self.pooling == 'sum':
            return h.sum(dim=1)
        elif self.pooling == 'mean':
            return h.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_points, 3)
        
        Returns:
            Output tensor of shape (batch, out_dim)
        """
        # Per-point encoding
        h = self.phi(x)  # (batch, n_points, phi_out_dim)
        
        # Pooling
        h = self._pool(h)  # (batch, phi_out_dim)
        
        # Set-level decoding
        h = self.rho(h)  # (batch, rho_out_dim)
        
        return h
    
    def forward_with_intermediate(self, x, layer_idx):
        """
        Forward pass that returns intermediate activations at specified layer.
        
        Args:
            x: Input tensor of shape (batch, n_points, 3)
            layer_idx: Layer index (1-based):
                - 1 to num_phi_layers: phi layers (per-point, shape (batch, n_points, H))
                - num_phi_layers + 1: after pooling (shape (batch, H))
                - num_phi_layers + 2 onwards: rho layers (shape (batch, H'))
                - -1: final output
        
        Returns:
            Activation tensor at the specified layer
        """
        if layer_idx == -1:
            return self.forward(x)
        
        if layer_idx < 1 or layer_idx > self.total_layers:
            raise ValueError(
                f"Invalid layer_idx: {layer_idx}. Must be between 1 and {self.total_layers}, or -1"
            )
        
        # Check if target is in phi
        if layer_idx <= self.num_phi_linear_layers:
            h, found = self._apply_phi_with_intermediate(x, layer_idx)
            if found:
                return h
            raise RuntimeError("Failed to retrieve phi intermediate activation.")
        
        # Run full phi
        h, _ = self._apply_phi_with_intermediate(x, None)
        
        # Check if target is pooling layer
        if layer_idx == self.num_phi_linear_layers + 1:
            return self._pool(h)
        
        # Pool and check rho layers
        h = self._pool(h)
        rho_target = layer_idx - self.num_phi_linear_layers - 1
        h, found = self._apply_rho_with_intermediate(h, rho_target)
        
        if found:
            return h
        
        raise RuntimeError("Failed to retrieve intermediate activation.")
    
    def is_layer_before_pooling(self, layer_idx):
        """
        Check if a layer index is before pooling (in phi network).
        
        Args:
            layer_idx: Layer index (1-based)
        
        Returns:
            True if layer is in phi (before pooling), False otherwise
        """
        if layer_idx == -1:
            return False
        return layer_idx <= self.num_phi_linear_layers

