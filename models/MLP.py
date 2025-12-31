import torch
import torch.nn as nn


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
        