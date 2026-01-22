"""
Base protocol for symmetry metrics.

All metrics should implement this interface to be usable with the registry.
"""

from typing import Protocol, Dict, Any, runtime_checkable
import torch
import torch.nn as nn
from pathlib import Path


@runtime_checkable
class SymmetryMetric(Protocol):
    """
    Protocol for symmetry metrics.
    
    A symmetry metric measures how invariant a model's representations
    are to transformations from a symmetry group (e.g., SO(2), SO(3)).
    
    Implementing classes should:
    1. Define a `name` class attribute
    2. Implement `compute()` to calculate metric values per layer
    3. Implement `plot()` to visualize the results
    """
    
    name: str
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute metric for all layers in the model.
        
        Args:
            model: Neural network model with `forward_with_intermediate` method.
            data: Input data tensor of shape (N, input_dim).
            device: Torch device for computation.
            **kwargs: Metric-specific parameters.
        
        Returns:
            Dictionary mapping layer names to metric values.
            Example: {"layer_1": 0.95, "layer_2": 0.82, "output": 0.01}
        """
        ...
    
    def plot(
        self,
        values: Dict[str, float],
        save_path: Path = None,
        **kwargs
    ) -> None:
        """
        Plot metric values by layer.
        
        Args:
            values: Dictionary of layer names to metric values.
            save_path: Optional path to save the plot.
            **kwargs: Plot-specific parameters.
        """
        ...


class BaseMetric:
    """
    Base class for symmetry metrics with common functionality.
    
    Provides default implementations that can be overridden.
    """
    
    name: str = "base"
    
    def __init__(self, **kwargs):
        """Store any metric-specific parameters."""
        self.params = kwargs
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")
    
    def plot(
        self,
        values: Dict[str, float],
        save_path: Path = None,
        **kwargs
    ) -> None:
        """Default bar plot implementation."""
        import matplotlib.pyplot as plt
        
        layers = list(values.keys())
        vals = list(values.values())
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(layers))
        ax.bar(x, vals, color='steelblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_xlabel('Layer')
        ax.set_ylabel(f'{self.name}')
        ax.set_title(f'{self.name} by Layer')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
