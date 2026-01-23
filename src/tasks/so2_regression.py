"""
SO(2) Regression Task: Scalar field prediction on a 2D disk.

This task tests whether networks learn SO(2)-invariant representations
when the target function depends only on radius (invariant) vs. when
it depends on angle (non-invariant).

Data:
    - Points sampled uniformly from a disk/annulus
    - Full SO(2) rotational symmetry in the sampling
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional


class Field(ABC):
    """Abstract base class for scalar fields on a 2D disk."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Evaluate the scalar field at given points.
        
        Args:
            x: X coordinates array.
            y: Y coordinates array.
            r: Radii array.
        
        Returns:
            Scalar field values at each point.
        """
        pass


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


class GaussianRing(Field):
    """SO(2)-invariant Gaussian ring field: exp(-(r-center)²/(2*std²))."""
    
    def __init__(self, center: float = 0.6, std: float = 0.08):
        self.center = center
        self.std = std
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.exp(-((r - self.center) ** 2) / (2 * self.std ** 2))


class XField(Field):
    """Non-invariant x-coordinate field: f(x,y) = x."""
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return x


class Fourier(Field):
    """Fourier mode field: cos(k*θ). NOT SO(2)-invariant for k != 0."""
    
    def __init__(self, k: int = 2):
        self.k = k
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        theta = np.arctan2(y, x)
        return np.cos(self.k * theta)


class Mix(Field):
    """Mixed field: (1-α)*field1 + α*field2. Interpolates between any two fields."""
    
    def __init__(self, alpha: float = 0.5, field1: Field = None, field2: Field = None):
        self.alpha = alpha
        self.field1 = field1 if field1 is not None else GaussianRing()
        self.field2 = field2 if field2 is not None else Fourier(k=2)
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        part1 = self.field1(x, y, r)
        part2 = self.field2(x, y, r)
        return (1 - self.alpha) * part1 + self.alpha * part2


# =============================================================================
# SO(2) INVARIANT FIELDS (depend only on radius r)
# =============================================================================

class Constant(Field):
    """Trivially SO(2)-invariant constant field: f(x,y) = c."""
    
    def __init__(self, c: float = 1.0):
        self.c = c
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.full_like(r, self.c)


class Radius(Field):
    """SO(2)-invariant linear radius field: f(x,y) = r."""
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return r


class RadiusPower(Field):
    """SO(2)-invariant polynomial radius field: f(x,y) = r^p."""
    
    def __init__(self, p: float = 2.0):
        self.p = p
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return r ** self.p


class RadialSine(Field):
    """SO(2)-invariant radial oscillation: f(x,y) = sin(k*π*r)."""
    
    def __init__(self, k: float = 2.0):
        self.k = k
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.sin(self.k * np.pi * r)


class MultiRing(Field):
    """SO(2)-invariant sum of Gaussian rings at different radii."""
    
    def __init__(self, centers: list = None, std: float = 0.08):
        self.centers = centers if centers is not None else [0.3, 0.7]
        self.std = std
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        result = np.zeros_like(r)
        for c in self.centers:
            result += np.exp(-((r - c) ** 2) / (2 * self.std ** 2))
        return result


class RadialStep(Field):
    """SO(2)-invariant step function: f(x,y) = 1 if r > threshold else 0."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return (r > self.threshold).astype(np.float64)


class BesselJ0(Field):
    """SO(2)-invariant Bessel function: f(x,y) = J₀(scale*r)."""
    
    def __init__(self, scale: float = 10.0):
        self.scale = scale
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        from scipy.special import j0
        return j0(self.scale * r)


class RadialGabor(Field):
    """SO(2)-invariant localized radial oscillation: exp(-(r-c)²/2σ²) * cos(freq*r)."""
    
    def __init__(self, center: float = 0.5, std: float = 0.15, freq: float = 8.0):
        self.center = center
        self.std = std
        self.freq = freq
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        envelope = np.exp(-((r - self.center) ** 2) / (2 * self.std ** 2))
        oscillation = np.cos(self.freq * r)
        return envelope * oscillation


# =============================================================================
# NON-SO(2) INVARIANT FIELDS (depend on angle θ)
# =============================================================================

class YField(Field):
    """Non-invariant y-coordinate field: f(x,y) = y."""
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return y


class Quadrant(Field):
    """Non-invariant checkerboard pattern: f(x,y) = sign(x)*sign(y)."""
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.sign(y)


class Spiral(Field):
    """Non-invariant spiral wave: f(x,y) = sin(θ + rate*r)."""
    
    def __init__(self, rate: float = 1.0):
        self.rate = rate
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        theta = np.arctan2(y, x)
        return np.sin(theta + self.rate * r)


class RadialAngular(Field):
    """Non-invariant product of radial and angular: f(x,y) = r * cos(k*θ)."""
    
    def __init__(self, k: int = 1):
        self.k = k
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        theta = np.arctan2(y, x)
        return r * np.cos(self.k * theta)


class Dipole(Field):
    """Non-invariant dipole-like field: f(x,y) = x / (r + eps)."""
    
    def __init__(self, eps: float = 0.01):
        self.eps = eps
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        return x / (r + self.eps)


class Vortex(Field):
    """Non-invariant angle field: f(x,y) = θ / π (normalized to [-1, 1])."""
    
    def __call__(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        theta = np.arctan2(y, x)
        return theta / np.pi


class ScalarFieldDataset(Dataset):
    """
    Dataset for scalar field regression on a 2D disk.
    
    Points are sampled uniformly from a disk with full rotational symmetry.
    Target is computed using a scalar field.
    
    Attributes:
        X: Input coordinates of shape (n_samples, 2).
        y: Target values of shape (n_samples, 1).
        r_min, r_max: Radius bounds.
        field: The Field instance used.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        r_min: float = 0.0,
        r_max: float = 1.0,
        scalar_field_fn: Optional[Field] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_samples: Total number of samples.
            r_min: Minimum radius.
            r_max: Maximum radius.
            scalar_field_fn: Field instance to evaluate. Defaults to GaussianRing().
            seed: Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.r_min = r_min
        self.r_max = r_max
        self.seed = seed
        
        if scalar_field_fn is None:
            scalar_field_fn = GaussianRing()
        
        self.scalar_field_fn = scalar_field_fn
        
        rng = np.random.default_rng(seed)
        
        # Sample points uniformly from disk
        x, y, r = sample_uniform_disk(n_samples, r_min, r_max, rng)
        
        # Compute scalar field target using the provided field
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
    scalar_field_fn: Optional[Field] = None,
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
        scalar_field_fn: Field instance to evaluate. Defaults to GaussianRing().
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
        scalar_field_fn=scalar_field_fn,
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
