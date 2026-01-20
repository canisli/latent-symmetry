"""
Data generation for SO(2)-invariant scalar field regression task.

Points are sampled uniformly over a disk with full rotational symmetry.
Target is computed using a scalar field function (e.g., Gaussian)."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Callable


def sample_uniform_disk(
    n: int, 
    r_min: float = 0.0, 
    r_max: float = 3.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample n points uniformly from an annulus/disk with SO(2) symmetry.
    
    Uses the inverse CDF method for uniform sampling over annular area:
    r = sqrt(r_min² + u * (r_max² - r_min²)) where u ~ Unif[0,1]
    θ ~ Unif[0, 2π)
    
    Args:
        n: Number of points to sample.
        r_min: Minimum radius (0 for full disk).
        r_max: Maximum radius.
        rng: Random number generator.
    
    Returns:
        Tuple of (x, y, r) coordinates and radii as numpy arrays.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Uniform sampling over annular area
    u = rng.uniform(0, 1, n)
    r = np.sqrt(r_min**2 + u * (r_max**2 - r_min**2))
    
    # Uniform angle (full SO(2) symmetry)
    theta = rng.uniform(0, 2 * np.pi, n)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return x, y, r


def gaussian_ring(x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    SO(2)-invariant Gaussian ring: exp(-(r-0.6)² / (2*0.08²)).
    
    Args:
        x: X coordinates (unused, for interface compatibility).
        y: Y coordinates (unused, for interface compatibility).
        r: Array of radii.
    
    Returns:
        Array of Gaussian values.
    """
    center, std = 0.6, 0.08
    return np.exp(-((r - center) ** 2) / (2 * std ** 2))


def x_field(x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Non-invariant scalar field: f(x,y) = x.
    
    This is NOT SO(2)-invariant since rotation changes x.
    
    Args:
        x: X coordinates.
        y: Y coordinates (unused).
        r: Array of radii (unused).
    
    Returns:
        Array equal to x coordinates.
    """
    return x


class ScalarFieldDataset(Dataset):
    """
    SO(2)-invariant dataset for scalar field regression.
    
    Points are sampled uniformly from a disk with full rotational symmetry.
    Target is computed using a scalar field function (e.g., Gaussian).
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        r_min: float = 0.0,
        r_max: float = 1.0,
        scalar_field_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_samples: Total number of samples.
            r_min: Minimum radius.
            r_max: Maximum radius.
            scalar_field_fn: Function that takes (x, y, r) arrays and returns scalar field values.
                           Defaults to gaussian_ring.
            seed: Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.r_min = r_min
        self.r_max = r_max
        self.seed = seed
        
        if scalar_field_fn is None:
            scalar_field_fn = gaussian_ring
        
        self.scalar_field_fn = scalar_field_fn
        
        rng = np.random.default_rng(seed)
        
        # Sample points uniformly from disk
        x, y, r = sample_uniform_disk(n_samples, r_min, r_max, rng)
        
        # Compute scalar field target using the provided function
        target = scalar_field_fn(x, y, r)
        
        # Store as tensors
        self.X = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return data as numpy arrays for plotting."""
        return self.X.numpy(), self.y.numpy().squeeze()


def create_dataloaders(
    n_samples: int = 1000,
    r_min: float = 0.0,
    r_max: float = 1.0,
    scalar_field_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
    train_split: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, ScalarFieldDataset]:
    """
    Create train and validation dataloaders for scalar field regression.
    
    Args:
        n_samples: Total number of samples.
        r_min: Minimum radius.
        r_max: Maximum radius.
        scalar_field_fn: Function that takes (x, y, r) arrays and returns scalar field values.
                       Defaults to gaussian_ring.
        train_split: Fraction of data for training.
        batch_size: Batch size for dataloaders.
        seed: Random seed for reproducibility.
        num_workers: Number of workers for dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, full_dataset).
    """
    # Create full dataset
    dataset = ScalarFieldDataset(
        n_samples=n_samples, 
        r_min=r_min, 
        r_max=r_max,
        scalar_field_fn=scalar_field_fn or gaussian_ring,
        seed=seed
    )
    
    # Split into train/val
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    
    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    return train_loader, val_loader, dataset


