import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims: list[int], act=nn.ReLU, aux_head_layer: int = None):
        """
        Args:
            dims: List of layer sizes, including input and output.
                  Example: [3, 128, 128, 1]
            act: Activation class to use between linear layers (default: ReLU).
            aux_head_layer: Optional layer index (1-based) to attach an auxiliary head.
                  If specified, creates a linear projection from that layer's output
                  dimension to the final output dimension.
        """
        super().__init__()
        
        self.dims = dims
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
            if i < len(dims) - 1:
                layers.append(act())

        self.layers = nn.Sequential(*layers)
        self.num_linear_layers = len(dims) - 1
        
        # Create auxiliary head if specified
        self.aux_head_layer = aux_head_layer
        self.aux_head = None
        if aux_head_layer is not None:
            if aux_head_layer < 1 or aux_head_layer >= self.num_linear_layers:
                raise ValueError(
                    f"aux_head_layer must be between 1 and {self.num_linear_layers - 1}, "
                    f"got {aux_head_layer}"
                )
            # dims[aux_head_layer] is the output dimension of that layer
            hidden_dim = dims[aux_head_layer]
            output_dim = dims[-1]
            self.aux_head = nn.Linear(hidden_dim, output_dim)

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

    def forward_aux_head(self, x):
        """
        Forward pass through the auxiliary head.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
        
        Returns:
            Auxiliary head output tensor of shape (batch_size, output_dim)
        
        Raises:
            RuntimeError: If no auxiliary head is configured
        """
        if self.aux_head is None:
            raise RuntimeError("No auxiliary head configured. Set aux_head_layer in __init__.")
        
        # Get intermediate activations at the auxiliary head layer
        h = self.forward_with_intermediate(x, self.aux_head_layer)
        # Apply the auxiliary head
        return self.aux_head(h)

