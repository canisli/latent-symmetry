"""
Task definitions for symmetry experiments.

Each task module defines:
- Field base class and implementations
- Data sampling functions
- Dataset class
- Dataloader factory
"""

from .so2_regression import (
    sample_uniform_disk,
    Field,
    # Original fields
    GaussianRing,
    XField,
    Fourier,
    Mix,
    # SO(2) invariant fields
    Constant,
    Radius,
    RadiusPower,
    RadialSine,
    MultiRing,
    RadialStep,
    BesselJ0,
    RadialGabor,
    # Non-SO(2) invariant fields
    YField,
    Quadrant,
    Spiral,
    RadialAngular,
    Dipole,
    Vortex,
    # Dataset and utilities
    ScalarFieldDataset,
    create_dataloaders,
)

__all__ = [
    "sample_uniform_disk",
    "Field",
    # Original fields
    "GaussianRing",
    "XField",
    "Fourier",
    "Mix",
    # SO(2) invariant fields
    "Constant",
    "Radius",
    "RadiusPower",
    "RadialSine",
    "MultiRing",
    "RadialStep",
    "BesselJ0",
    "RadialGabor",
    # Non-SO(2) invariant fields
    "YField",
    "Quadrant",
    "Spiral",
    "RadialAngular",
    "Dipole",
    "Vortex",
    # Dataset and utilities
    "ScalarFieldDataset",
    "create_dataloaders",
]
