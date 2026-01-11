"""
Dataset for convex hull volume prediction task.

The convex hull volume of a set of 3D points is SO(3) invariant -
rotating all points by the same rotation preserves the volume.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial import ConvexHull


def compute_convex_hull_volume(points):
    """
    Compute the volume of the convex hull of a set of 3D points.
    
    Args:
        points: numpy array of shape (n_points, 3)
    
    Returns:
        volume: float, the volume of the convex hull
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Degenerate case (e.g., coplanar points)
        return None


class ConvexHullDataset(Dataset):
    """
    Dataset of point sets with their convex hull volumes.
    
    Each sample consists of:
    - points: (n_points, 3) tensor of 3D coordinates
    - volume: scalar tensor with the convex hull volume
    """
    
    def __init__(self, n_samples, n_points=50, seed=None, extent=1.0):
        """
        Args:
            n_samples: Number of samples to generate
            n_points: Number of points per sample (default: 50)
            seed: Random seed for reproducibility
            extent: Points are sampled uniformly in [-extent, extent]^3 cube
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.n_points = n_points
        self.extent = extent
        
        # Generate all samples
        points_list = []
        volumes_list = []
        
        generated = 0
        max_attempts = n_samples * 2  # Allow some retries for degenerate cases
        attempts = 0
        
        while generated < n_samples and attempts < max_attempts:
            # Sample points uniformly in cube
            points = np.random.uniform(-extent, extent, size=(n_points, 3))
            
            # Compute convex hull volume
            volume = compute_convex_hull_volume(points)
            
            if volume is not None and volume > 1e-10:  # Skip degenerate cases
                points_list.append(points)
                volumes_list.append(volume)
                generated += 1
            
            attempts += 1
        
        if generated < n_samples:
            raise RuntimeError(f"Could only generate {generated}/{n_samples} valid samples")
        
        # Convert to tensors
        self.points = torch.tensor(np.stack(points_list), dtype=torch.float32)
        self.volumes = torch.tensor(np.array(volumes_list), dtype=torch.float32).unsqueeze(-1)
        
        # Compute statistics for reference
        self.volume_mean = self.volumes.mean().item()
        self.volume_std = self.volumes.std().item()
    
    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        return self.points[idx], self.volumes[idx]
    
    def get_flattened(self, idx):
        """
        Get a sample with flattened points for MLP input.
        
        Returns:
            points_flat: (n_points * 3,) tensor
            volume: (1,) tensor
        """
        points, volume = self[idx]
        return points.reshape(-1), volume


class ConvexHullDatasetFlattened(Dataset):
    """
    Wrapper that returns flattened points for MLP input.
    """
    
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: ConvexHullDataset instance
        """
        self.base = base_dataset
        self.n_points = base_dataset.n_points
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        points, volume = self.base[idx]
        return points.reshape(-1), volume


if __name__ == '__main__':
    # Test the dataset
    print("Testing ConvexHullDataset...")
    
    dataset = ConvexHullDataset(n_samples=1000, n_points=50, seed=42)
    print(f"Dataset size: {len(dataset)}")
    print(f"Points shape: {dataset.points.shape}")
    print(f"Volumes shape: {dataset.volumes.shape}")
    print(f"Volume mean: {dataset.volume_mean:.4f}")
    print(f"Volume std: {dataset.volume_std:.4f}")
    print(f"Volume range: [{dataset.volumes.min().item():.4f}, {dataset.volumes.max().item():.4f}]")
    
    # Test single sample
    points, volume = dataset[0]
    print(f"\nSingle sample:")
    print(f"  Points shape: {points.shape}")
    print(f"  Volume: {volume.item():.4f}")
    
    # Test flattened wrapper
    flat_dataset = ConvexHullDatasetFlattened(dataset)
    points_flat, volume = flat_dataset[0]
    print(f"\nFlattened sample:")
    print(f"  Points shape: {points_flat.shape}")
    print(f"  Volume: {volume.item():.4f}")
    
    # Verify SO(3) invariance
    print("\nVerifying SO(3) invariance...")
    from symmetry import sample_so3_rotation
    
    points, volume = dataset[0]
    R = sample_so3_rotation(1)[0]  # Single rotation matrix
    
    # Rotate all points
    points_rotated = (R @ points.T).T
    
    # Compute volumes
    vol_original = compute_convex_hull_volume(points.numpy())
    vol_rotated = compute_convex_hull_volume(points_rotated.numpy())
    
    print(f"  Original volume: {vol_original:.6f}")
    print(f"  Rotated volume:  {vol_rotated:.6f}")
    print(f"  Difference:      {abs(vol_original - vol_rotated):.2e}")

