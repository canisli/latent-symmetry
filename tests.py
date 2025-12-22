import torch
from train import set_seed
from models import MLP, sample_so3_rotation
from data import ScalarFieldDataset


def test_mlp():
    set_seed(42)
    mlp = MLP(1, [10], 5)
    assert mlp(torch.rand((100, 1))).shape == (100, 5)


def test_sample_so3_rotation_uniformity():
    """
    Test that sample_so3_rotation produces uniform (Haar) rotations on SO(3).
    
    We verify:
    1. Matrices are valid SO(3): orthogonal (R @ R^T = I) and det(R) = 1
    2. Distribution is uniform: rotating a fixed vector yields uniform distribution
       on S^2, which has zero mean and variance 1/3 in each coordinate.
    """
    set_seed(42)
    n_samples = 50000
    
    R = sample_so3_rotation(n_samples)
    
    # Test 1: Check orthogonality (R @ R^T = I)
    RRT = torch.bmm(R, R.transpose(1, 2))
    identity = torch.eye(3).unsqueeze(0).expand(n_samples, -1, -1)
    ortho_error = (RRT - identity).abs().max()
    assert ortho_error < 1e-5, f"Orthogonality error too large: {ortho_error}"
    
    # Test 2: Check determinant = 1 (proper rotations, not reflections)
    dets = torch.linalg.det(R)
    det_error = (dets - 1.0).abs().max()
    assert det_error < 1e-5, f"Determinant error too large: {det_error}"
    
    # Test 3: Check uniformity on S^2
    # Apply rotations to a fixed unit vector; result should be uniform on sphere
    v = torch.tensor([[1.0, 0.0, 0.0]]).expand(n_samples, -1)
    v_rotated = torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)  # (n_samples, 3)
    
    # For uniform distribution on unit sphere:
    # - E[x] = E[y] = E[z] = 0
    # - E[x^2] = E[y^2] = E[z^2] = 1/3
    mean = v_rotated.mean(dim=0)
    variance = (v_rotated ** 2).mean(dim=0)
    
    # Use tolerance based on CLT: std of sample mean ~ 1/sqrt(n)
    mean_tol = 5 / (n_samples ** 0.5)  # ~0.022 for n=50000
    var_tol = 5 / (n_samples ** 0.5)
    
    assert mean.abs().max() < mean_tol, f"Mean not close to zero: {mean}"
    assert (variance - 1/3).abs().max() < var_tol, f"Variance not close to 1/3: {variance}"
    
    # Test 4: Check that different fixed vectors give same statistics (isotropy)
    v2 = torch.tensor([[0.0, 1.0, 0.0]]).expand(n_samples, -1)
    v2_rotated = torch.bmm(R, v2.unsqueeze(-1)).squeeze(-1)
    mean2 = v2_rotated.mean(dim=0)
    variance2 = (v2_rotated ** 2).mean(dim=0)
    
    assert mean2.abs().max() < mean_tol, f"Mean (y-axis) not close to zero: {mean2}"
    assert (variance2 - 1/3).abs().max() < var_tol, f"Variance (y-axis) not close to 1/3: {variance2}"