"""
Symmetry metrics package.

Provides modular metrics for measuring symmetry properties in neural networks.

Usage:
    from latsym.metrics import get_metric, list_metrics
    
    # List available metrics
    print(list_metrics())  # ['Q', ...]
    
    # Get a metric instance
    metric = get_metric("Q", n_rotations=32)
    
    # Compute and plot
    values = metric.compute(model, data, device=device)
    metric.plot(values, save_path="q_vs_layer.png")
"""

from .registry import get_metric, list_metrics, register, is_registered
from .base import SymmetryMetric, BaseMetric

# Import metric modules to trigger registration
from . import q_metric

__all__ = [
    # Registry functions
    "get_metric",
    "list_metrics",
    "register",
    "is_registered",
    # Base classes
    "SymmetryMetric",
    "BaseMetric",
]
