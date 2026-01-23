"""
Task definitions for symmetry experiments.

Each task module defines:
- Data sampling functions
- Target functions (invariant and non-invariant)
- Dataset class
- Dataloader factory
"""

from .so2_regression import (
    sample_uniform_disk,
    gaussian_ring,
    x_field,
    fourier,
    mix,
    ScalarFieldDataset,
    create_dataloaders,
)

__all__ = [
    "sample_uniform_disk",
    "gaussian_ring",
    "x_field",
    "fourier",
    "mix",
    "ScalarFieldDataset",
    "create_dataloaders",
]
