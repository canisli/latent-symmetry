"""
Seed management utilities for reproducible experiments.

Provides deterministic seed derivation for different randomness sources
(data, model, augmentation) from a single experiment seed.
"""

import hashlib
import os
import random
import numpy as np
import torch


def derive_seed(run_seed: int, category: str) -> int:
    """
    Deterministically derive a seed from a base seed and category name.
    
    Args:
        run_seed: Base seed for the experiment run
        category: Category name (e.g., "data", "model", "augmentation")
    
    Returns:
        Derived seed value
    """
    # Use hashlib for deterministic derivation across Python sessions
    # Python's built-in hash() is non-deterministic due to hash randomization
    seed_str = f"{run_seed}_{category}"
    seed_bytes = seed_str.encode('utf-8')
    hash_obj = hashlib.md5(seed_bytes)
    hash_int = int(hash_obj.hexdigest(), 16)

    return hash_int % (2**31)


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for full reproducibility.
    
    This should be called at the VERY START of the program, before any
    other operations that might use random number generators.
    
    This sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch's CPU and CUDA random generators
    - CuDNN deterministic mode
    - PyTorch deterministic algorithms
    
    Args:
        seed: Seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable PyTorch deterministic mode (raises error on non-deterministic ops)
    # Set CUBLAS_WORKSPACE_CONFIG for CUDA >= 10.2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_model_seed(seed: int) -> None:
    """
    Set random seeds for model initialization (weights, dropout, etc.).
    
    This is a lighter version that just sets RNG state without enabling
    deterministic algorithms (which can be slow).
    
    Args:
        seed: Seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
