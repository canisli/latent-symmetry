import torch
from torch.utils.data import DataLoader
from train import set_model_seed, derive_seed
from models import MLP
from symmetry import sample_so3_rotation
from data import ScalarFieldDataset


def test_mlp():
    set_model_seed(42)
    mlp = MLP([1, 10, 5])
    assert mlp(torch.rand((100, 1))).shape == (100, 5)


def test_forward_with_intermediate():
    model = MLP([2, 3, 4, 1])
    
    # hardcode linear layers
    with torch.no_grad():
        model.layers[0].weight.data = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        model.layers[0].bias.data = torch.tensor([0.1, 0.2, 0.3])
        
        model.layers[2].weight.data = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        model.layers[2].bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
        
        model.layers[4].weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        model.layers[4].bias.data = torch.tensor([0.5])
    
    # Test input
    x = torch.tensor([[1.0, 2.0]])
    
    # Manual computation for verification
    # Layer 1: Linear(2 -> 3) + ReLU
    h1_linear = torch.matmul(x, model.layers[0].weight.t()) + model.layers[0].bias
    # h1_linear = [1*1 + 2*2 + 0.1, 1*3 + 2*4 + 0.2, 1*5 + 2*6 + 0.3]
    #           = [1 + 4 + 0.1, 3 + 8 + 0.2, 5 + 12 + 0.3]
    #           = [5.1, 11.2, 17.3]
    h1_expected = torch.relu(h1_linear)
    
    # Layer 2: Linear(3 -> 4) + ReLU
    h2_linear = torch.matmul(h1_expected, model.layers[2].weight.t()) + model.layers[2].bias
    h2_expected = torch.relu(h2_linear)
    
    # Layer 3: Linear(4 -> 1) (final output, no activation)
    h3_expected = torch.matmul(h2_expected, model.layers[4].weight.t()) + model.layers[4].bias
    
    # Test layer_idx=1 (after first activation)
    result1 = model.forward_with_intermediate(x, layer_idx=1)
    assert torch.allclose(result1, h1_expected, atol=1e-6), \
        f"Layer 1 mismatch: expected {h1_expected}, got {result1}"
    
    # Test layer_idx=2 (after second activation)
    result2 = model.forward_with_intermediate(x, layer_idx=2)
    assert torch.allclose(result2, h2_expected, atol=1e-6), \
        f"Layer 2 mismatch: expected {h2_expected}, got {result2}"
    
    # Test layer_idx=-1 (final output)
    result_final = model.forward_with_intermediate(x, layer_idx=-1)
    assert torch.allclose(result_final, h3_expected, atol=1e-6), \
        f"Final output mismatch: expected {h3_expected}, got {result_final}"
    
    # Verify that forward() gives same result as layer_idx=-1
    forward_result = model.forward(x)
    assert torch.allclose(forward_result, result_final, atol=1e-6), \
        "forward() should match forward_with_intermediate(..., -1)"
    
    # Test with batch of size 2
    x_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result1_batch = model.forward_with_intermediate(x_batch, layer_idx=1)
    # Manually compute for batch
    h1_batch_linear = torch.matmul(x_batch, model.layers[0].weight.t()) + model.layers[0].bias
    h1_batch_expected = torch.relu(h1_batch_linear)
    assert torch.allclose(result1_batch, h1_batch_expected, atol=1e-6), \
        "Batch processing failed for layer_idx=1"


def test_weight_initialization_reproducibility():
    """
    Test that weight initialization is reproducible with the same seed.
    Creates two models with the same architecture and seed, and verifies
    that all weights are identical.
    """
    seed = 42
    
    # Create first model
    set_model_seed(seed)
    model1 = MLP([3, 128, 128, 128, 1])
    
    # Create second model with same seed
    set_model_seed(seed)
    model2 = MLP([3, 128, 128, 128, 1])
    
    # Check that all weights are identical
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
        assert torch.allclose(param1, param2, atol=1e-7), f"Weights differ for {name1}"
    
    # Also verify that different seeds produce different weights
    set_model_seed(seed + 1)
    model3 = MLP([3, 128, 128, 128, 1])
    
    # At least one weight should differ
    weights_differ = False
    for (name1, param1), (name3, param3) in zip(model1.named_parameters(), model3.named_parameters()):
        if not torch.allclose(param1, param3, atol=1e-7):
            weights_differ = True
            break
    
    assert weights_differ, "Models with different seeds should have different weights"


def test_sample_so3_rotation_uniformity():
    """
    Test that sample_so3_rotation produces uniform (Haar) rotations on SO(3).
    
    We verify:
    1. Matrices are valid SO(3): orthogonal (R @ R^T = I) and det(R) = 1
    2. Distribution is uniform: rotating a fixed vector yields uniform distribution
       on S^2, which has zero mean and variance 1/3 in each coordinate.
    """
    set_model_seed(42)
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


def test_data_seed_reproducibility():
    """
    Test that data seed produces reproducible datasets, splits, and DataLoader order.
    """
    seed = 42
    n_samples = 1000
    
    # Create first dataset with same seed
    dataset1 = ScalarFieldDataset(n_samples, seed=seed)
    
    # Create second dataset with same seed
    dataset2 = ScalarFieldDataset(n_samples, seed=seed)
    
    # Verify datasets are identical
    assert torch.allclose(dataset1.X, dataset2.X), "Datasets should be identical with same seed"
    assert torch.allclose(dataset1.y, dataset2.y), "Dataset targets should be identical with same seed"
    
    # Test train/test/val split reproducibility
    generator1 = torch.Generator().manual_seed(seed)
    generator2 = torch.Generator().manual_seed(seed)
    train_ds1, val_ds1, test_ds1 = torch.utils.data.random_split(dataset1, [0.6, 0.2, 0.2], generator=generator1)
    train_ds2, val_ds2, test_ds2 = torch.utils.data.random_split(dataset2, [0.6, 0.2, 0.2], generator=generator2)
    
    # Verify splits are identical (check indices)
    assert train_ds1.indices == train_ds2.indices, "Train splits should be identical"
    assert val_ds1.indices == val_ds2.indices, "Val splits should be identical"
    assert test_ds1.indices == test_ds2.indices, "Test splits should be identical"
    
    # Test DataLoader shuffling reproducibility
    loader_gen1 = torch.Generator().manual_seed(seed)
    loader_gen2 = torch.Generator().manual_seed(seed)
    loader1 = DataLoader(train_ds1, batch_size=32, shuffle=True, generator=loader_gen1)
    loader2 = DataLoader(train_ds2, batch_size=32, shuffle=True, generator=loader_gen2)
    
    # Get first batch from each loader
    batch1_x, batch1_y = next(iter(loader1))
    batch2_x, batch2_y = next(iter(loader2))
    
    # Verify batches are identical (same order)
    assert torch.allclose(batch1_x, batch2_x), "First batches should be identical with same seed"
    assert torch.allclose(batch1_y, batch2_y), "First batch targets should be identical with same seed"
    
    # Verify that different seeds produce different datasets
    dataset3 = ScalarFieldDataset(n_samples, seed=seed + 1)
    datasets_differ = not torch.allclose(dataset1.X, dataset3.X, atol=1e-7)
    assert datasets_differ, "Datasets with different seeds should differ"


def test_augmentation_seed_reproducibility():
    """
    Test that augmentation seed produces reproducible SO(3) rotations.
    """
    seed = 42
    batch_size = 100
    
    # Create generators with same seed
    generator1 = torch.Generator().manual_seed(seed)
    generator2 = torch.Generator().manual_seed(seed)
    
    # Sample rotations with same generator
    R1 = sample_so3_rotation(batch_size, generator=generator1)
    R2 = sample_so3_rotation(batch_size, generator=generator2)
    
    # Verify rotations are identical
    assert torch.allclose(R1, R2, atol=1e-7), "Rotations should be identical with same generator seed"
    
    # Test multiple calls with same generator (should produce different rotations)
    generator3 = torch.Generator().manual_seed(seed)
    R3_first = sample_so3_rotation(batch_size, generator=generator3)
    R3_second = sample_so3_rotation(batch_size, generator=generator3)
    
    # These should be different (generator advances state)
    rotations_differ = not torch.allclose(R3_first, R3_second, atol=1e-7)
    assert rotations_differ, "Consecutive calls with same generator should produce different rotations"
    
    # Verify that different seeds produce different rotations
    generator4 = torch.Generator().manual_seed(seed + 1)
    R4 = sample_so3_rotation(batch_size, generator=generator4)
    
    rotations_differ = not torch.allclose(R1, R4, atol=1e-7)
    assert rotations_differ, "Rotations with different seeds should differ"
    
    # Test derive_seed function produces consistent seeds
    seed1 = derive_seed(42, "augmentation")
    seed2 = derive_seed(42, "augmentation")
    assert seed1 == seed2, "derive_seed should be deterministic"
    
    # Test that different categories produce different seeds
    data_seed = derive_seed(42, "data")
    model_seed = derive_seed(42, "model")
    aug_seed = derive_seed(42, "augmentation")
    
    assert data_seed != model_seed, "Data and model seeds should differ"
    assert data_seed != aug_seed, "Data and augmentation seeds should differ"
    assert model_seed != aug_seed, "Model and augmentation seeds should differ"