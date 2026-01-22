"""
Metric registry for auto-discovery and instantiation.

Metrics register themselves using the @register decorator.
"""

from typing import Dict, Type, List
from .base import SymmetryMetric


# Global registry of metric classes
_METRICS: Dict[str, Type[SymmetryMetric]] = {}


def register(name: str):
    """
    Decorator to register a metric class.
    
    Usage:
        @register("Q")
        class QMetric(BaseMetric):
            ...
    
    Args:
        name: Name to register the metric under.
    
    Returns:
        Decorator function.
    """
    def decorator(cls: Type[SymmetryMetric]) -> Type[SymmetryMetric]:
        if name in _METRICS:
            raise ValueError(f"Metric '{name}' is already registered")
        _METRICS[name] = cls
        # Also set the name on the class if not already set
        if not hasattr(cls, 'name') or cls.name == "base":
            cls.name = name
        return cls
    return decorator


def get_metric(name: str, **kwargs) -> SymmetryMetric:
    """
    Get an instance of a registered metric.
    
    Args:
        name: Name of the metric to get.
        **kwargs: Parameters to pass to the metric constructor.
    
    Returns:
        Instance of the metric.
    
    Raises:
        KeyError: If metric is not registered.
    """
    if name not in _METRICS:
        available = ", ".join(_METRICS.keys()) or "(none)"
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")
    return _METRICS[name](**kwargs)


def list_metrics() -> List[str]:
    """
    List all registered metric names.
    
    Returns:
        List of registered metric names.
    """
    return list(_METRICS.keys())


def is_registered(name: str) -> bool:
    """Check if a metric is registered."""
    return name in _METRICS
