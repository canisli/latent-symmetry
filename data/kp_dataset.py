import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from energyflow import EFPSet
import energyflow as ef

def compute_kps(
    fourmomenta: np.ndarray,          # (N, M, 4) in (E,px,py,pz)
    edges_list: List[List[Tuple[int,int]]],
    measure: str = 'kinematic',
    coords: str = 'epxpypz',
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
) -> np.ndarray:
    """
    Returns Y with shape (N, K) where K=len(edges_list).
    Truncates each event to its non-padded particles before calling EFPSet.
    
    For measure='kinematic': Computes Lorentz-invariant kinematic polynomials 
        using Mandelstam invariants (beta/kappa/normed are ignored).
    For measure='eeefm': Computes e+e- EFPs which are NOT Lorentz invariant
        (only rotationally invariant). Uses beta, kappa, and normed parameters.
        
    Args:
        normed: If True, normalize energies so sum(z_i) = 1. Default False
                to preserve scale information for non-invariant targets.
    """
    fm = np.asarray(fourmomenta, dtype=np.float32)
    if fm.ndim != 3 or fm.shape[-1] != 4:
        raise ValueError('fourmomenta must have shape (N, M, 4)')

    # mask of real (non-zero) four-vectors: (N, M)
    mask = np.any(fm != 0.0, axis=2)

    # Build EFPSet with appropriate parameters
    # Note: kinematic measure ignores beta/kappa/normed, but eeefm uses them
    kps = EFPSet(
        *edges_list, measure=measure, coords=coords, beta=beta, kappa=kappa, normed=normed
    )

    out = np.zeros((fm.shape[0], len(edges_list)), dtype=np.float32)
    for i in range(fm.shape[0]):
        part = fm[i, mask[i]]  # (n_i, 4)
        if part.shape[0] == 0:
            continue
        out[i] = kps.compute(part).astype(np.float32)

    return out


class KPSurrogateDataset(Dataset):
    """
    X: (N, M, 4) padded four-momenta
    Y: (N, 1, K) KP targets (singleton time/channel axis for easy broadcasting)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 2:  # (N, K) -> (N, 1, K)
            Y = Y[:, None, :]
        assert X.ndim == 3 and X.shape[-1] == 4, X.shape
        assert Y.ndim == 3, Y.shape
        assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
        self.X = X
        self.Y = Y

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]



def make_kp_dataloader(
    edges_list: List[List[Tuple[int,int]]], 
    n_events: int = 10_000,
    n_particles: int = 128,
    batch_size: int = 256,
    measure: str = 'kinematic',
    coords: str = 'epxpypz',
    input_scale: float = 1.0,
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
    target_transform: str = 'log1p',
):
    """
    Generate a DataLoader for kinematic polynomial surrogate training.
    
    input_scale rescales the four-momenta globally to stabilize training.
    
    Args:
        edges_list: List of edge configurations for EFPs
        n_events: Number of events to generate
        n_particles: Number of particles per event
        batch_size: Batch size for DataLoader
        measure: 'kinematic' (Lorentz invariant) or 'eeefm' (non-invariant)
        coords: Coordinate system ('epxpypz')
        input_scale: Scale factor for input four-momenta
        beta: Angular weighting exponent (used by eeefm, ignored by kinematic)
        kappa: Energy weighting exponent (used by eeefm, ignored by kinematic)
        normed: If True, normalize energies so sum(z_i) = 1 (default False)
        target_transform: How to transform targets:
            - 'log1p': log(1+x), good for KPs (compresses large values)
            - 'log_standardized': log(x) then z-score normalize (preserves relative diffs)
            - 'standardized': z-score normalize raw values (linear, preserves structure)
    """
    
    # Generate synthetic events as (E,px,py,pz) 
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    # Compute KP/EFP targets on UNSCALED momenta
    Y = compute_kps(X, edges_list, measure=measure, coords=coords, beta=beta, kappa=kappa, normed=normed)
    
    # Apply target transformation
    if target_transform == 'log1p':
        # Standard: log(1+x), compresses large differences
        Y = np.log1p(Y)
    elif target_transform == 'log_standardized':
        # log(x) then standardize - preserves relative differences better
        Y = np.log(Y + 1e-10)  # small epsilon to avoid log(0)
        Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-10)
    elif target_transform == 'standardized':
        # Raw standardization - fully preserves relative structure
        Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-10)
    else:
        raise ValueError(f"Unknown target_transform: {target_transform}")
    
    # Scale inputs AFTER computing targets (for numerical stability in the model)
    denom = float(input_scale) if input_scale not in (0, None) else 1.0
    X = X / denom
    
    # Create dataset and dataloader
    dataset = KPSurrogateDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return dataloader


def estimate_input_scale(n_events: int, n_particles: int, seed: Optional[int] = None) -> float:
    """
    Estimate a global scale for four-momenta so inputs land near unit std.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    scale = float(np.std(X))
    # Avoid degeneracy
    return scale if scale > 1e-12 else 1.0


def estimate_target_scale(
    edges_list: List[List[Tuple[int,int]]],
    n_events: int,
    n_particles: int,
    measure: str = 'kinematic',
    coords: str = 'epxpypz',
    beta: float = 2.0,
    kappa: float = 1.0,
    normed: bool = False,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate the standard deviation of raw (pre-log1p) target values.
    
    This is useful for understanding the scale difference between different
    measures (e.g., kinematic vs eeefm).
    
    Returns:
        Standard deviation of raw target values
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    Y = compute_kps(X, edges_list, measure=measure, coords=coords, beta=beta, kappa=kappa, normed=normed)
    
    scale = float(np.std(Y))
    return scale if scale > 1e-12 else 1.0