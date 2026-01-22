"""
Symmetry group implementations.

Each group module provides:
- transform(x, g): Apply group element g to data x
- sample(n, device): Sample n random group elements
- identity(device): Return identity element
"""

from .so2 import rotate, sample_rotations, rotation_matrix

__all__ = [
    # SO(2) - 2D rotations
    "rotate",
    "sample_rotations", 
    "rotation_matrix",
]
