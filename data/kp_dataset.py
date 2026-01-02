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
) -> np.ndarray:
    """
    Returns Y with shape (N, K) where K=len(edges_list).
    Truncates each event to its non-padded particles before calling EFPSet.
    Computes kinematic polynomials using Mandelstam invariants.
    """
    fm = np.asarray(fourmomenta, dtype=np.float32)
    if fm.ndim != 3 or fm.shape[-1] != 4:
        raise ValueError('fourmomenta must have shape (N, M, 4)')

    # mask of real (non-zero) four-vectors: (N, M)
    mask = np.any(fm != 0.0, axis=2)

    kps = EFPSet(
        *edges_list, measure=measure, coords=coords
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
):
    """
    Generate a DataLoader for kinematic polynomial surrogate training.
    
    input_scale rescales the four-momenta globally to stabilize training.
    Targets are log1p-transformed KP values computed on UNSCALED momenta.
    """
    
    # Generate synthetic events as (E,px,py,pz) 
    X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
    
    # Compute KP targets on UNSCALED momenta (KPs scale as p^k, so must use original scale)
    Y = compute_kps(X, edges_list, measure=measure, coords=coords)
    Y = np.log1p(Y)
    
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