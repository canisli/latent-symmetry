"""
MI (Mutual Information) metric for measuring SO(2) invariance.

MI_l = I(Δ; S_l) / H(Δ)

where:
- Δ = (θ_2 - θ_1) mod 2π is the relative rotation
- S_l = (r_l(x, θ_1), r_l(x, θ_2)) is the pair of representations

Estimated via a trained classifier:
- M̂_l = (log K - L_l) / log K
- L_l is the test cross-entropy of predicting Δ from (r_1, r_2)

- MI ≈ 0: No decodable rotation info (invariant)
- MI ≈ 1: Full rotation info retained (equivariant)

This metric is invariant to invertible nonlinear reparameterizations of the
representation, unlike energy-based metrics like Q.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .base import BaseMetric
from .registry import register
from .plotting import plot_metric_vs_layer, TrainingInfo
from ..groups.so2 import rotate


def get_discrete_angles(K: int, device: torch.device = None) -> torch.Tensor:
    """
    Get K uniformly spaced angles in [0, 2π).
    
    Args:
        K: Number of discrete angles.
        device: Torch device.
    
    Returns:
        Tensor of shape (K,) with angles.
    """
    return torch.linspace(0, 2 * math.pi, K + 1, device=device)[:-1]


class DeltaClassifier(nn.Module):
    """
    MLP classifier to predict relative rotation Δ from representation pairs.
    
    Input: concatenation of (r_1, r_2) where each r_i is a layer representation.
    Output: logits over K classes (discrete relative rotations).
    """
    
    def __init__(
        self,
        input_dim: int,
        K: int,
        hidden_dims: List[int] = [256, 256],
    ):
        """
        Args:
            input_dim: Dimension of each representation r_i.
            K: Number of discrete angle classes.
            hidden_dims: List of hidden layer dimensions.
        """
        super().__init__()
        
        # Input is concatenation of r_1 and r_2
        dims = [2 * input_dim] + hidden_dims + [K]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            r1: First representation, shape (batch, input_dim).
            r2: Second representation, shape (batch, input_dim).
        
        Returns:
            Logits over K classes, shape (batch, K).
        """
        x = torch.cat([r1, r2], dim=-1)
        return self.network(x)


def generate_mi_dataset(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    K: int,
    n_pairs_per_point: int = 4,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Generate dataset for training the MI classifier.
    
    For each data point x, sample n_pairs_per_point pairs of (θ_1, θ_2)
    from the discrete angle set Θ_K, compute representations, and record
    the relative rotation class.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        K: Number of discrete angles.
        n_pairs_per_point: Number of angle pairs per data point.
        device: Torch device.
    
    Returns:
        Tuple of (r1, r2, labels, repr_dim) where:
        - r1: First representations, shape (N * n_pairs, repr_dim)
        - r2: Second representations, shape (N * n_pairs, repr_dim)
        - labels: Class labels (relative rotation index), shape (N * n_pairs,)
        - repr_dim: Dimension of the representations
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model.to(device)
    data = data.to(device)
    N = data.shape[0]
    
    # Get discrete angles
    angles = get_discrete_angles(K, device=device)
    
    # Sample angle indices
    idx1 = torch.randint(0, K, (N, n_pairs_per_point), device=device)
    idx2 = torch.randint(0, K, (N, n_pairs_per_point), device=device)
    
    # Compute relative rotation class: (idx2 - idx1) mod K
    delta_idx = (idx2 - idx1) % K
    
    # Flatten
    idx1_flat = idx1.reshape(-1)  # (N * n_pairs,)
    idx2_flat = idx2.reshape(-1)
    delta_flat = delta_idx.reshape(-1)
    
    # Get actual angles
    theta1 = angles[idx1_flat]  # (N * n_pairs,)
    theta2 = angles[idx2_flat]
    
    # Expand data to match pairs
    data_expanded = data.unsqueeze(1).expand(-1, n_pairs_per_point, -1)
    data_flat = data_expanded.reshape(-1, 2)  # (N * n_pairs, 2)
    
    # Rotate data
    x_rot1 = rotate(data_flat, theta1)
    x_rot2 = rotate(data_flat, theta2)
    
    # Get representations
    with torch.no_grad():
        r1 = model.forward_with_intermediate(x_rot1, layer_idx)
        r2 = model.forward_with_intermediate(x_rot2, layer_idx)
    
    repr_dim = r1.shape[1]
    
    return r1, r2, delta_flat, repr_dim


def train_delta_classifier(
    r1_train: torch.Tensor,
    r2_train: torch.Tensor,
    labels_train: torch.Tensor,
    K: int,
    repr_dim: int,
    hidden_dims: List[int] = [256, 256],
    n_steps: int = 1000,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device = None,
) -> DeltaClassifier:
    """
    Train a classifier to predict relative rotation from representation pairs.
    
    Args:
        r1_train: Training first representations.
        r2_train: Training second representations.
        labels_train: Training labels (relative rotation class).
        K: Number of classes.
        repr_dim: Dimension of representations.
        hidden_dims: Classifier hidden layer dimensions.
        n_steps: Number of training steps.
        lr: Learning rate.
        batch_size: Batch size.
        device: Torch device.
    
    Returns:
        Trained DeltaClassifier.
    """
    if device is None:
        device = torch.device('cpu')
    
    classifier = DeltaClassifier(repr_dim, K, hidden_dims).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    # Create dataloader
    dataset = TensorDataset(r1_train, r2_train, labels_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    classifier.train()
    step = 0
    while step < n_steps:
        for r1_batch, r2_batch, labels_batch in loader:
            if step >= n_steps:
                break
            
            optimizer.zero_grad()
            logits = classifier(r1_batch, r2_batch)
            loss = F.cross_entropy(logits, labels_batch)
            loss.backward()
            optimizer.step()
            step += 1
    
    return classifier


def evaluate_classifier(
    classifier: DeltaClassifier,
    r1_test: torch.Tensor,
    r2_test: torch.Tensor,
    labels_test: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """
    Evaluate classifier and return mean cross-entropy loss.
    
    Args:
        classifier: Trained DeltaClassifier.
        r1_test: Test first representations.
        r2_test: Test second representations.
        labels_test: Test labels.
        batch_size: Batch size for evaluation.
    
    Returns:
        Mean cross-entropy loss on test set.
    """
    classifier.eval()
    
    dataset = TensorDataset(r1_test, r2_test, labels_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for r1_batch, r2_batch, labels_batch in loader:
            logits = classifier(r1_batch, r2_batch)
            # Use sum reduction to accumulate total loss
            loss = F.cross_entropy(logits, labels_batch, reduction='sum')
            total_loss += loss.item()
            total_samples += labels_batch.shape[0]
    
    return total_loss / total_samples


def normalize_representations(
    r1: torch.Tensor,
    r2: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize representations to zero mean and unit variance.
    
    This makes the MI estimator invariant to affine transformations,
    which is important since true MI is invariant under bijections.
    
    Statistics are computed jointly over r1 and r2 to preserve relative
    information between the pair.
    
    Args:
        r1: First representations, shape (N, D).
        r2: Second representations, shape (N, D).
        eps: Small constant for numerical stability.
    
    Returns:
        Tuple of normalized (r1, r2).
    """
    # Compute statistics over all representations jointly
    all_r = torch.cat([r1, r2], dim=0)  # (2N, D)
    mean = all_r.mean(dim=0, keepdim=True)
    std = all_r.std(dim=0, keepdim=True)
    
    # Normalize
    r1_norm = (r1 - mean) / (std + eps)
    r2_norm = (r2 - mean) / (std + eps)
    
    return r1_norm, r2_norm


def compute_MI(
    model: nn.Module,
    data: torch.Tensor,
    layer_idx: int,
    K: int = 16,
    n_pairs_per_point: int = 16,
    classifier_hidden: List[int] = [512, 512, 256],
    classifier_steps: int = 3000,
    lr: float = 1e-3,
    batch_size: int = 256,
    test_fraction: float = 0.2,
    device: torch.device = None,
    permutation_null: bool = False,
) -> float:
    """
    Compute MI metric for a single layer.
    
    M̂_l = (log K - L_l) / log K
    
    where L_l is the test cross-entropy of a classifier predicting Δ from (r_1, r_2).
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        layer_idx: Layer index (1-based for hidden, -1 for output).
        K: Number of discrete angles.
        n_pairs_per_point: Number of angle pairs per data point.
        classifier_hidden: Hidden layer dimensions for classifier.
        classifier_steps: Number of classifier training steps.
        lr: Classifier learning rate.
        batch_size: Batch size for classifier training/evaluation.
        test_fraction: Fraction of data to use for testing.
        device: Torch device.
        permutation_null: If True, shuffle labels to get null baseline.
    
    Returns:
        MI value for the layer (0 = invariant, 1 = equivariant).
    """
    if device is None:
        device = torch.device('cpu')
    
    # Generate dataset
    r1, r2, labels, repr_dim = generate_mi_dataset(
        model, data, layer_idx, K, n_pairs_per_point, device
    )
    
    # Normalize representations to make estimator scale/offset invariant
    r1, r2 = normalize_representations(r1, r2)
    
    # Permutation null: shuffle labels
    if permutation_null:
        perm = torch.randperm(labels.shape[0], device=device)
        labels = labels[perm]
    
    # Train/test split
    n_samples = r1.shape[0]
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test
    
    # Shuffle indices
    indices = torch.randperm(n_samples, device=device)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    r1_train, r2_train = r1[train_idx], r2[train_idx]
    labels_train = labels[train_idx]
    r1_test, r2_test = r1[test_idx], r2[test_idx]
    labels_test = labels[test_idx]
    
    # Train classifier
    classifier = train_delta_classifier(
        r1_train, r2_train, labels_train,
        K, repr_dim, classifier_hidden,
        classifier_steps, lr, batch_size, device
    )
    
    # Evaluate
    L_l = evaluate_classifier(classifier, r1_test, r2_test, labels_test, batch_size)
    
    # Compute normalized MI
    log_K = math.log(K)
    MI = (log_K - L_l) / log_K
    
    # Clamp to [0, 1] (can be slightly negative due to estimation noise)
    MI = max(0.0, min(1.0, MI))
    
    return MI


def compute_all_MI(
    model: nn.Module,
    data: torch.Tensor,
    K: int = 16,
    n_pairs_per_point: int = 16,
    classifier_hidden: List[int] = [512, 512, 256],
    classifier_steps: int = 3000,
    lr: float = 1e-3,
    batch_size: int = 256,
    test_fraction: float = 0.2,
    device: torch.device = None,
    permutation_null: bool = False,
) -> Dict[str, float]:
    """
    Compute MI for all layers in the model.
    
    Args:
        model: Neural network model.
        data: Input data tensor of shape (N, 2).
        K: Number of discrete angles.
        n_pairs_per_point: Number of angle pairs per data point.
        classifier_hidden: Hidden layer dimensions for classifier.
        classifier_steps: Number of classifier training steps.
        lr: Classifier learning rate.
        batch_size: Batch size.
        test_fraction: Fraction of data for testing.
        device: Torch device.
        permutation_null: If True, shuffle labels for null baseline.
    
    Returns:
        Dictionary mapping layer names to MI values.
    """
    MI_values = {}
    
    # Hidden layers (1-indexed)
    for layer_idx in range(1, model.num_linear_layers):
        MI = compute_MI(
            model, data, layer_idx, K, n_pairs_per_point,
            classifier_hidden, classifier_steps, lr, batch_size,
            test_fraction, device, permutation_null
        )
        MI_values[f'layer_{layer_idx}'] = MI
    
    # Output layer
    MI_out = compute_MI(
        model, data, -1, K, n_pairs_per_point,
        classifier_hidden, classifier_steps, lr, batch_size,
        test_fraction, device, permutation_null
    )
    MI_values['output'] = MI_out
    
    return MI_values


def compute_oracle_MI(
    data: torch.Tensor,
    targets: torch.Tensor,
    scalar_field_fn,
    K: int = 16,
    n_pairs_per_point: int = 16,
    classifier_hidden: List[int] = [512, 512, 256],
    classifier_steps: int = 3000,
    lr: float = 1e-3,
    batch_size: int = 256,
    test_fraction: float = 0.2,
    device: torch.device = None,
) -> float:
    """
    Compute MI for the oracle (perfect predictor where ŷ = y).
    
    This evaluates what MI would be at the output if the model perfectly
    predicted the true labels. For invariant targets, oracle MI ≈ 0.
    For non-invariant targets, oracle MI > 0.
    
    Args:
        data: Input data tensor of shape (N, 2).
        targets: True target values of shape (N, 1).
        scalar_field_fn: Function (x, y, r) -> target used to compute labels for rotated points.
        K: Number of discrete angles.
        n_pairs_per_point: Number of angle pairs per data point.
        classifier_hidden: Hidden layer dimensions for classifier.
        classifier_steps: Number of classifier training steps.
        lr: Classifier learning rate.
        batch_size: Batch size.
        test_fraction: Fraction of data for testing.
        device: Torch device.
    
    Returns:
        Oracle MI value.
    """
    import numpy as np
    
    if device is None:
        device = torch.device('cpu')
    
    data = data.to(device)
    N = data.shape[0]
    
    # Get discrete angles
    angles = get_discrete_angles(K, device=device)
    
    # Sample angle indices
    idx1 = torch.randint(0, K, (N, n_pairs_per_point), device=device)
    idx2 = torch.randint(0, K, (N, n_pairs_per_point), device=device)
    
    # Compute relative rotation class: (idx2 - idx1) mod K
    delta_idx = (idx2 - idx1) % K
    
    # Flatten
    idx1_flat = idx1.reshape(-1)
    idx2_flat = idx2.reshape(-1)
    delta_flat = delta_idx.reshape(-1)
    
    # Get actual angles
    theta1 = angles[idx1_flat]
    theta2 = angles[idx2_flat]
    
    # Expand data to match pairs
    data_expanded = data.unsqueeze(1).expand(-1, n_pairs_per_point, -1)
    data_flat = data_expanded.reshape(-1, 2)
    
    # Rotate data
    x_rot1 = rotate(data_flat, theta1)
    x_rot2 = rotate(data_flat, theta2)
    
    # Compute true labels for rotated points (these are the "representations")
    x1_np, y1_np = x_rot1[:, 0].cpu().numpy(), x_rot1[:, 1].cpu().numpy()
    x2_np, y2_np = x_rot2[:, 0].cpu().numpy(), x_rot2[:, 1].cpu().numpy()
    r1_np = np.sqrt(x1_np**2 + y1_np**2)
    r2_np = np.sqrt(x2_np**2 + y2_np**2)
    
    y_rot1 = torch.tensor(scalar_field_fn(x1_np, y1_np, r1_np), dtype=torch.float32, device=device).unsqueeze(1)
    y_rot2 = torch.tensor(scalar_field_fn(x2_np, y2_np, r2_np), dtype=torch.float32, device=device).unsqueeze(1)
    
    # Use true labels as "representations"
    r1, r2 = y_rot1, y_rot2
    repr_dim = 1
    
    # Normalize representations to make estimator scale/offset invariant
    r1, r2 = normalize_representations(r1, r2)
    
    # Train/test split
    n_samples = r1.shape[0]
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test
    
    indices = torch.randperm(n_samples, device=device)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    r1_train, r2_train = r1[train_idx], r2[train_idx]
    labels_train = delta_flat[train_idx]
    r1_test, r2_test = r1[test_idx], r2[test_idx]
    labels_test = delta_flat[test_idx]
    
    # Train classifier
    classifier = train_delta_classifier(
        r1_train, r2_train, labels_train,
        K, repr_dim, classifier_hidden,
        classifier_steps, lr, batch_size, device
    )
    
    # Evaluate
    L_l = evaluate_classifier(classifier, r1_test, r2_test, labels_test, batch_size)
    
    # Compute normalized MI
    log_K = math.log(K)
    MI = (log_K - L_l) / log_K
    MI = max(0.0, min(1.0, MI))
    
    return MI


def plot_mi_vs_layer(
    mi_values: Dict[str, float],
    save_path: Path = None,
    oracle_MI: float = None,
    run_name: str = None,
    sym_penalty_type: str = None,
    sym_layers: list = None,
    lambda_sym: float = 0.0,
    null_values: Dict[str, float] = None,
):
    """
    Plot MI as a function of layer depth.
    
    Args:
        mi_values: Dictionary mapping layer names to MI values.
        save_path: Optional path to save the plot.
        oracle_MI: Optional oracle MI value to show as additional bar.
        run_name: Optional run name (unused, kept for API compatibility).
        sym_penalty_type: Type of symmetry penalty used during training.
        sym_layers: List of layers penalized during training.
        lambda_sym: Lambda value for symmetry penalty.
        null_values: Optional permutation null values for comparison.
    """
    training_info = TrainingInfo(
        penalty_type=sym_penalty_type,
        layers=sym_layers,
        lambda_sym=lambda_sym,
    )
    plot_metric_vs_layer(
        values=mi_values,
        metric_name='MI',
        save_path=save_path,
        color='darkorange',
        ylabel='MI (Mutual Information)',
        oracle_value=oracle_MI,
        training_info=training_info,
    )


@register("MI")
class MIMetric(BaseMetric):
    """
    MI (Mutual Information) metric for measuring SO(2) invariance.
    
    MI_l = I(Δ; S_l) / H(Δ)
    
    Estimated via a trained classifier:
    M̂_l = (log K - L_l) / log K
    
    where L_l is the test cross-entropy of predicting the relative rotation Δ
    from pairs of representations (r_l(x, θ_1), r_l(x, θ_2)).
    
    - MI ≈ 0: No decodable rotation info (invariant)
    - MI ≈ 1: Full rotation info retained (equivariant)
    
    This metric is invariant to invertible nonlinear reparameterizations,
    unlike energy-based metrics like Q.
    
    Parameters:
        K: Number of discrete angles (default: 16)
        n_pairs_per_point: Angle pairs per data point (default: 16)
        classifier_hidden: Classifier hidden dims (default: [512, 512, 256])
        classifier_steps: Training steps (default: 3000)
        lr: Learning rate (default: 1e-3)
        batch_size: Batch size (default: 256)
        test_fraction: Test set fraction (default: 0.2)
    """
    
    name = "MI"
    
    def __init__(
        self,
        K: int = 16,
        n_pairs_per_point: int = 16,
        classifier_hidden: List[int] = [512, 512, 256],
        classifier_steps: int = 3000,
        lr: float = 1e-3,
        batch_size: int = 256,
        test_fraction: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.K = K
        self.n_pairs_per_point = n_pairs_per_point
        self.classifier_hidden = classifier_hidden
        self.classifier_steps = classifier_steps
        self.lr = lr
        self.batch_size = batch_size
        self.test_fraction = test_fraction
    
    def compute(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute MI for all layers."""
        return compute_all_MI(
            model,
            data,
            K=kwargs.get('K', self.K),
            n_pairs_per_point=kwargs.get('n_pairs_per_point', self.n_pairs_per_point),
            classifier_hidden=kwargs.get('classifier_hidden', self.classifier_hidden),
            classifier_steps=kwargs.get('classifier_steps', self.classifier_steps),
            lr=kwargs.get('lr', self.lr),
            batch_size=kwargs.get('batch_size', self.batch_size),
            test_fraction=kwargs.get('test_fraction', self.test_fraction),
            device=device,
            permutation_null=kwargs.get('permutation_null', False),
        )
    
    def plot(
        self,
        values: Dict[str, float],
        save_path: Path = None,
        **kwargs
    ) -> None:
        """Plot MI values."""
        run_name = kwargs.get('run_name', None)
        null_values = kwargs.get('null_values', None)
        plot_mi_vs_layer(values, save_path, run_name=run_name, null_values=null_values)
