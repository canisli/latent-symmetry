"""Unit tests verifying that kinematic polynomials are Lorentz invariant."""

import numpy as np
import torch
import pytest
import energyflow as ef

from data.kp_dataset import compute_kps
from symmetry import rand_lorentz, rand_rotation, rand_boost


# Common graph structures for testing
EDGES_LIST = [
    [(0, 1)],                    # 2-point correlator
    [(0, 1), (1, 2)],            # 3-point chain
    [(0, 1), (1, 2), (0, 2)],    # Triangle
]


def assert_nontrivial_transform(original: np.ndarray, transformed: np.ndarray, msg: str, atol: float = 1e-8) -> None:
    """Guard against identity transforms that would make invariance tests vacuous."""
    if np.allclose(original, transformed, rtol=0.0, atol=atol):
        raise AssertionError(msg)


def apply_lorentz_numpy(fourmomenta: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Apply Lorentz transformation L to four-momenta.
    
    Args:
        fourmomenta: Array of shape (N, M, 4) or (M, 4)
        L: Lorentz transformation matrix of shape (4, 4) or (N, 4, 4)
    
    Returns:
        Transformed four-momenta with same shape as input.
    """
    if fourmomenta.ndim == 2:
        # Single event: (M, 4) @ (4, 4).T -> (M, 4)
        return fourmomenta @ L.T
    elif fourmomenta.ndim == 3:
        if L.ndim == 2:
            # Batch of events with single transform: (N, M, 4) @ (4, 4).T
            return fourmomenta @ L.T
        else:
            # Batch of events with batch of transforms: (N, M, 4) @ (N, 4, 4).T
            # einsum: 'nmi,nji->nmj' but we want L @ p, so 'nij,nmj->nmi'
            return np.einsum('nij,nmj->nmi', L, fourmomenta)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {fourmomenta.ndim}D")


class TestKPLorentzInvariance:
    """Tests verifying that KPs are invariant under Lorentz transformations."""

    @pytest.fixture
    def sample_events(self):
        """Generate sample events for testing."""
        np.random.seed(42)
        n_events = 10
        n_particles = 8
        X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
        return X

    def test_kp_invariant_under_rotation(self, sample_events):
        """Test that KPs are invariant under spatial rotations."""
        torch.manual_seed(123)
        
        # Compute original KPs
        kps_original = compute_kps(sample_events, EDGES_LIST)
        
        # Generate random rotation matrix
        R = rand_rotation(
            shape=torch.Size([]),
            dtype=torch.float64,
        ).numpy()
        
        # Apply rotation to all events
        rotated_events = apply_lorentz_numpy(sample_events, R)
        assert_nontrivial_transform(sample_events, rotated_events, "Rotation unexpectedly leaves events unchanged")
        
        # Compute KPs on rotated events
        kps_rotated = compute_kps(rotated_events, EDGES_LIST)
        
        # KPs should be invariant
        np.testing.assert_allclose(
            kps_original, kps_rotated,
            rtol=1e-4, atol=1e-6,
            err_msg="KPs are not invariant under rotation"
        )

    def test_kp_invariant_under_boost(self, sample_events):
        """Test that KPs are invariant under Lorentz boosts."""
        torch.manual_seed(456)
        
        # Compute original KPs
        kps_original = compute_kps(sample_events, EDGES_LIST)
        
        # Generate random boost (use moderate rapidity for numerical stability)
        B = rand_boost(
            shape=torch.Size([]),
            std_eta=0.3,
            dtype=torch.float64,
        ).numpy()
        
        # Apply boost to all events
        boosted_events = apply_lorentz_numpy(sample_events, B)
        assert_nontrivial_transform(sample_events, boosted_events, "Boost unexpectedly leaves events unchanged")
        
        # Compute KPs on boosted events
        kps_boosted = compute_kps(boosted_events, EDGES_LIST)
        
        # KPs should be invariant
        np.testing.assert_allclose(
            kps_original, kps_boosted,
            rtol=1e-4, atol=1e-6,
            err_msg="KPs are not invariant under boost"
        )

    def test_kp_invariant_under_general_lorentz(self, sample_events):
        """Test that KPs are invariant under general Lorentz transformations."""
        torch.manual_seed(789)
        
        # Compute original KPs
        kps_original = compute_kps(sample_events, EDGES_LIST)
        
        # Generate random Lorentz transformation (rotation * boost)
        L = rand_lorentz(
            shape=torch.Size([]),
            std_eta=0.3,
            dtype=torch.float64,
        ).numpy()
        
        # Apply transformation to all events
        transformed_events = apply_lorentz_numpy(sample_events, L)
        assert_nontrivial_transform(sample_events, transformed_events, "Lorentz transform unexpectedly leaves events unchanged")
        
        # Compute KPs on transformed events
        kps_transformed = compute_kps(transformed_events, EDGES_LIST)
        
        # KPs should be invariant
        np.testing.assert_allclose(
            kps_original, kps_transformed,
            rtol=1e-4, atol=1e-6,
            err_msg="KPs are not invariant under general Lorentz transformation"
        )

    def test_kp_invariant_under_multiple_lorentz_transforms(self, sample_events):
        """Test KP invariance under multiple different Lorentz transforms."""
        np.random.seed(42)
        n_transforms = 5
        
        # Compute original KPs
        kps_original = compute_kps(sample_events, EDGES_LIST)
        
        for i in range(n_transforms):
            torch.manual_seed(1000 + i)
            
            L = rand_lorentz(
                shape=torch.Size([]),
                std_eta=0.5,
                dtype=torch.float64,
            ).numpy()
            
            transformed_events = apply_lorentz_numpy(sample_events, L)
            assert_nontrivial_transform(sample_events, transformed_events, f"Lorentz transform {i} unexpectedly leaves events unchanged")
            kps_transformed = compute_kps(transformed_events, EDGES_LIST)
            
            np.testing.assert_allclose(
                kps_original, kps_transformed,
                rtol=1e-4, atol=1e-6,
                err_msg=f"KPs not invariant under Lorentz transform {i}"
            )

    def test_kp_invariant_per_event_transforms(self):
        """Test KP invariance when each event gets a different Lorentz transform."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_events = 5
        n_particles = 6
        
        # Generate sample events
        X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
        
        # Compute original KPs
        kps_original = compute_kps(X, EDGES_LIST)
        
        # Generate different Lorentz transform for each event
        L = rand_lorentz(
            shape=torch.Size([n_events]),
            std_eta=0.3,
            dtype=torch.float64,
        ).numpy()
        
        # Apply per-event transforms
        transformed = apply_lorentz_numpy(X, L)
        assert_nontrivial_transform(X, transformed, "Per-event Lorentz transform unexpectedly leaves events unchanged")
        
        # Compute KPs on transformed events
        kps_transformed = compute_kps(transformed, EDGES_LIST)
        
        # KPs should be invariant for each event
        np.testing.assert_allclose(
            kps_original, kps_transformed,
            rtol=1e-4, atol=1e-6,
            err_msg="KPs not invariant under per-event Lorentz transforms"
        )

    def test_kp_invariant_with_different_graphs(self):
        """Test KP invariance for various graph structures."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_events = 5
        n_particles = 6
        
        # Test different graph structures
        graph_structures = [
            [[(0, 1)]],                              # Single edge
            [[(0, 1), (1, 2)]],                      # Chain
            [[(0, 1), (1, 2), (0, 2)]],              # Triangle
            [[(0, 1)], [(0, 1), (1, 2)]],            # Multiple graphs
            [[(0, 1), (2, 3)]],                      # Disconnected edges
        ]
        
        X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
        
        L = rand_lorentz(
            shape=torch.Size([]),
            std_eta=0.4,
            dtype=torch.float64,
        ).numpy()
        
        X_transformed = apply_lorentz_numpy(X, L)
        assert_nontrivial_transform(X, X_transformed, "Lorentz transform for graph tests unexpectedly leaves events unchanged")
        
        for edges_list in graph_structures:
            kps_original = compute_kps(X, edges_list)
            kps_transformed = compute_kps(X_transformed, edges_list)
            
            np.testing.assert_allclose(
                kps_original, kps_transformed,
                rtol=1e-4, atol=1e-6,
                err_msg=f"KPs not invariant for graph {edges_list}"
            )

    def test_kp_invariant_large_boost(self):
        """Test KP invariance under larger boosts (stress test numerical stability)."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_events = 5
        n_particles = 4
        
        X = ef.gen_random_events_mcom(n_events, n_particles, dim=4).astype(np.float32)
        kps_original = compute_kps(X, EDGES_LIST)
        
        # Larger rapidity boost
        B = rand_boost(
            shape=torch.Size([]),
            std_eta=0.8,
            n_max_std_eta=2.0,
            dtype=torch.float64,
        ).numpy()
        
        X_boosted = apply_lorentz_numpy(X, B)
        assert_nontrivial_transform(X, X_boosted, "Large boost unexpectedly leaves events unchanged")
        kps_boosted = compute_kps(X_boosted, EDGES_LIST)
        
        # Same strict tolerance as other tests - KPs should be truly invariant
        np.testing.assert_allclose(
            kps_original, kps_boosted,
            rtol=1e-4, atol=1e-6,
            err_msg="KPs not invariant under large boost"
        )


class TestLorentzTransformValidity:
    """Tests to verify our Lorentz transforms are valid."""

    def test_rotation_preserves_lorentz_norm(self):
        """Rotations should preserve the Lorentz norm (mass squared)."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate sample four-momenta
        X = ef.gen_random_events_mcom(5, 4, dim=4).astype(np.float64)
        
        # Compute original Lorentz norms (m^2 = E^2 - p^2)
        norms_original = X[..., 0]**2 - np.sum(X[..., 1:]**2, axis=-1)
        
        R = rand_rotation(shape=torch.Size([]), dtype=torch.float64).numpy()
        X_rotated = apply_lorentz_numpy(X, R)
        
        norms_rotated = X_rotated[..., 0]**2 - np.sum(X_rotated[..., 1:]**2, axis=-1)
        
        # Note: gen_random_events_mcom generates massless particles, so norms are ~0
        # Use atol for comparing near-zero values
        np.testing.assert_allclose(norms_original, norms_rotated, rtol=1e-10, atol=1e-14)

    def test_boost_preserves_lorentz_norm(self):
        """Boosts should preserve the Lorentz norm (mass squared)."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        X = ef.gen_random_events_mcom(5, 4, dim=4).astype(np.float64)
        norms_original = X[..., 0]**2 - np.sum(X[..., 1:]**2, axis=-1)
        
        B = rand_boost(shape=torch.Size([]), std_eta=0.3, dtype=torch.float64).numpy()
        X_boosted = apply_lorentz_numpy(X, B)
        
        norms_boosted = X_boosted[..., 0]**2 - np.sum(X_boosted[..., 1:]**2, axis=-1)
        
        # Note: gen_random_events_mcom generates massless particles, so norms are ~0
        # Use atol for comparing near-zero values
        np.testing.assert_allclose(norms_original, norms_boosted, rtol=1e-10, atol=1e-14)

    def test_lorentz_transform_preserves_lorentz_norm(self):
        """General Lorentz transforms should preserve the Lorentz norm."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        X = ef.gen_random_events_mcom(5, 4, dim=4).astype(np.float64)
        norms_original = X[..., 0]**2 - np.sum(X[..., 1:]**2, axis=-1)
        
        L = rand_lorentz(shape=torch.Size([]), std_eta=0.3, dtype=torch.float64).numpy()
        X_transformed = apply_lorentz_numpy(X, L)
        
        norms_transformed = X_transformed[..., 0]**2 - np.sum(X_transformed[..., 1:]**2, axis=-1)
        
        # Note: gen_random_events_mcom generates massless particles, so norms are ~0
        # Use atol for comparing near-zero values
        np.testing.assert_allclose(norms_original, norms_transformed, rtol=1e-10, atol=1e-14)


class TestMandelstamInvariance:
    """Direct tests on Mandelstam invariants which underlie KPs."""

    def compute_mandelstam(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Compute Mandelstam invariant s_ij = (p_i + p_j)^2."""
        p_sum = p1 + p2
        # Using (+,-,-,-) metric: s = E^2 - |p|^2
        s = p_sum[..., 0]**2 - np.sum(p_sum[..., 1:]**2, axis=-1)
        return s

    def test_mandelstam_invariant_under_lorentz(self):
        """Mandelstam invariants should be Lorentz invariant."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate two particles
        event = ef.gen_random_events_mcom(1, 4, dim=4).astype(np.float64)[0]
        p1, p2 = event[0], event[1]
        
        s_original = self.compute_mandelstam(p1, p2)
        
        L = rand_lorentz(shape=torch.Size([]), std_eta=0.5, dtype=torch.float64).numpy()
        
        p1_t = (L @ p1.T).T
        p2_t = (L @ p2.T).T
        
        s_transformed = self.compute_mandelstam(p1_t, p2_t)
        
        np.testing.assert_allclose(s_original, s_transformed, rtol=1e-10)

    def test_all_pairwise_mandelstams_invariant(self):
        """All pairwise Mandelstam invariants should be preserved."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_particles = 5
        event = ef.gen_random_events_mcom(1, n_particles, dim=4).astype(np.float64)[0]
        
        # Compute all pairwise Mandelstams
        s_original = np.zeros((n_particles, n_particles))
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                s_original[i, j] = self.compute_mandelstam(event[i], event[j])
        
        L = rand_lorentz(shape=torch.Size([]), std_eta=0.5, dtype=torch.float64).numpy()
        event_t = apply_lorentz_numpy(event, L)
        
        s_transformed = np.zeros((n_particles, n_particles))
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                s_transformed[i, j] = self.compute_mandelstam(event_t[i], event_t[j])
        
        np.testing.assert_allclose(s_original, s_transformed, rtol=1e-10)

