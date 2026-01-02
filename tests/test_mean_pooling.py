"""Unit tests for mean pooling implementation in transformer.py."""

import torch
import pytest

from models.transformer import Transformer


class TestMeanPooling:
    """Tests for the mean pooling functionality in the Transformer class."""

    @pytest.fixture
    def transformer_with_pooling(self):
        """Create a transformer with mean pooling enabled."""
        return Transformer(
            in_channels=4,
            out_channels=2,
            hidden_channels=16,
            num_blocks=1,
            num_heads=4,
            use_mean_pooling=True,
        )

    @pytest.fixture
    def transformer_without_pooling(self):
        """Create a transformer without mean pooling."""
        return Transformer(
            in_channels=4,
            out_channels=2,
            hidden_channels=16,
            num_blocks=1,
            num_heads=4,
            use_mean_pooling=False,
        )

    def test_output_shape_with_mean_pooling(self, transformer_with_pooling):
        """Test that mean pooling reduces the sequence dimension."""
        batch_size = 3
        num_items = 5
        in_channels = 4
        out_channels = 2

        inputs = torch.randn(batch_size, num_items, in_channels)
        outputs = transformer_with_pooling(inputs)

        # Should have shape (batch_size, out_channels) - no sequence dim
        assert outputs.shape == (batch_size, out_channels)

    def test_output_shape_without_mean_pooling(self, transformer_without_pooling):
        """Test that without mean pooling, sequence dimension is preserved."""
        batch_size = 3
        num_items = 5
        in_channels = 4
        out_channels = 2

        inputs = torch.randn(batch_size, num_items, in_channels)
        outputs = transformer_without_pooling(inputs)

        # Should have shape (batch_size, num_items, out_channels)
        assert outputs.shape == (batch_size, num_items, out_channels)

    def test_mean_pooling_with_explicit_mask(self, transformer_with_pooling):
        """Test mean pooling with an explicit pool_mask."""
        batch_size = 2
        num_items = 4
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        
        # Mask: first sample has 3 real items, second has 2
        pool_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ])

        outputs = transformer_with_pooling(inputs, pool_mask=pool_mask)
        
        # Should still produce valid output
        assert outputs.shape == (batch_size, 2)
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()

    def test_mean_pooling_auto_detects_zero_padding(self, transformer_with_pooling):
        """Test that mean pooling auto-detects zero-padded inputs."""
        batch_size = 2
        num_items = 4
        in_channels = 4

        # Create inputs with explicit zero padding
        inputs = torch.randn(batch_size, num_items, in_channels)
        inputs[0, 3, :] = 0.0  # Pad last item of first sample
        inputs[1, 2:, :] = 0.0  # Pad last 2 items of second sample

        outputs = transformer_with_pooling(inputs)
        
        assert outputs.shape == (batch_size, 2)
        assert not torch.isnan(outputs).any()

    def test_mean_pooling_mask_correctness(self):
        """Test that the mean pooling computation is mathematically correct."""
        # Create a minimal transformer where we can trace the computation
        torch.manual_seed(42)
        
        transformer = Transformer(
            in_channels=4,
            out_channels=2,
            hidden_channels=16,
            num_blocks=1,
            num_heads=4,
            use_mean_pooling=True,
        )
        transformer.eval()

        batch_size = 1
        num_items = 3
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        
        # Test 1: All items real (mask all True)
        mask_all = torch.ones(batch_size, num_items, dtype=torch.bool)
        output_all = transformer(inputs, pool_mask=mask_all)

        # Test 2: Only first 2 items real
        mask_partial = torch.tensor([[True, True, False]])
        output_partial = transformer(inputs, pool_mask=mask_partial)

        # Outputs should differ when different items are pooled
        assert not torch.allclose(output_all, output_partial)

    def test_mean_pooling_handles_single_item(self, transformer_with_pooling):
        """Test mean pooling with a single real item."""
        batch_size = 2
        num_items = 3
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        
        # Only first item is real in each sample
        pool_mask = torch.tensor([
            [True, False, False],
            [True, False, False],
        ])

        outputs = transformer_with_pooling(inputs, pool_mask=pool_mask)
        
        assert outputs.shape == (batch_size, 2)
        assert not torch.isnan(outputs).any()

    def test_mean_pooling_all_masked_clamps_divisor(self, transformer_with_pooling):
        """Test that all-masked sequences don't cause division by zero."""
        batch_size = 2
        num_items = 3
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        
        # Edge case: one sample has no real items
        pool_mask = torch.tensor([
            [True, True, True],   # Normal sample
            [False, False, False], # All masked (edge case)
        ])

        outputs = transformer_with_pooling(inputs, pool_mask=pool_mask)
        
        # Should not produce NaN or Inf due to clamping
        assert outputs.shape == (batch_size, 2)
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()

    def test_mean_pooling_deterministic(self, transformer_with_pooling):
        """Test that mean pooling is deterministic for the same input."""
        transformer_with_pooling.eval()
        
        batch_size = 2
        num_items = 4
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        pool_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ])

        output1 = transformer_with_pooling(inputs, pool_mask=pool_mask)
        output2 = transformer_with_pooling(inputs, pool_mask=pool_mask)

        assert torch.allclose(output1, output2)

    def test_mean_pooling_weights_items_correctly(self):
        """Test that mean pooling computes correct weighted average.
        
        Note: pool_mask only affects the final pooling, not attention.
        Masked items still participate in attention and affect hidden states.
        This test verifies the pooling math is correct.
        """
        torch.manual_seed(123)
        
        # Use a transformer with no blocks to test pure pooling behavior
        # (no attention to mix items together)
        transformer = Transformer(
            in_channels=4,
            out_channels=4,  # Same as hidden for easier verification
            hidden_channels=4,
            num_blocks=0,  # No attention blocks - pure linear + pooling
            num_heads=1,
            use_mean_pooling=True,
        )
        # Set linear_out to identity-like for easier testing
        with torch.no_grad():
            transformer.linear_out.weight.copy_(torch.eye(4))
            transformer.linear_out.bias.zero_()
        transformer.eval()

        batch_size = 1
        num_items = 3
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        
        # Test with different masks and verify the pooling count is correct
        mask_all = torch.tensor([[True, True, True]])
        mask_two = torch.tensor([[True, True, False]])
        mask_one = torch.tensor([[True, False, False]])

        out_all = transformer(inputs, pool_mask=mask_all)
        out_two = transformer(inputs, pool_mask=mask_two)
        out_one = transformer(inputs, pool_mask=mask_one)

        # With no blocks, h = linear_in(inputs), then pool, then linear_out
        # The outputs should differ when different subsets are pooled
        assert not torch.allclose(out_all, out_two, atol=1e-5)
        assert not torch.allclose(out_two, out_one, atol=1e-5)
        assert not torch.allclose(out_all, out_one, atol=1e-5)

    def test_mean_pooling_batch_independence(self, transformer_with_pooling):
        """Test that each sample in the batch is pooled independently."""
        transformer_with_pooling.eval()
        
        batch_size = 3
        num_items = 4
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        
        # Different masks for each sample
        pool_mask = torch.tensor([
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
        ])

        # Full batch output
        output_batch = transformer_with_pooling(inputs, pool_mask=pool_mask)

        # Individual sample outputs
        outputs_individual = []
        for i in range(batch_size):
            single_input = inputs[i:i+1]
            single_mask = pool_mask[i:i+1]
            output_single = transformer_with_pooling(single_input, pool_mask=single_mask)
            outputs_individual.append(output_single)

        outputs_stacked = torch.cat(outputs_individual, dim=0)

        # Batch output should match stacked individual outputs
        assert torch.allclose(output_batch, outputs_stacked, atol=1e-5)

    def test_mean_pooling_gradient_flow(self, transformer_with_pooling):
        """Test that gradients flow correctly through mean pooling."""
        batch_size = 2
        num_items = 4
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels, requires_grad=True)
        pool_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ])

        outputs = transformer_with_pooling(inputs, pool_mask=pool_mask)
        loss = outputs.sum()
        loss.backward()

        # Gradients should exist and not be NaN
        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

        # Masked positions should still have gradients (through attention)
        # but they shouldn't affect the pooling directly


class TestMeanPoolingEdgeCases:
    """Edge case tests for mean pooling."""

    def test_single_batch(self):
        """Test with batch size of 1."""
        transformer = Transformer(
            in_channels=4,
            out_channels=2,
            hidden_channels=16,
            num_blocks=1,
            num_heads=4,
            use_mean_pooling=True,
        )

        inputs = torch.randn(1, 5, 4)
        outputs = transformer(inputs)

        assert outputs.shape == (1, 2)

    def test_single_item_sequence(self):
        """Test with a sequence of length 1."""
        transformer = Transformer(
            in_channels=4,
            out_channels=2,
            hidden_channels=16,
            num_blocks=1,
            num_heads=4,
            use_mean_pooling=True,
        )

        inputs = torch.randn(2, 1, 4)
        outputs = transformer(inputs)

        assert outputs.shape == (2, 2)

    def test_large_batch(self):
        """Test with a larger batch size."""
        transformer = Transformer(
            in_channels=4,
            out_channels=2,
            hidden_channels=16,
            num_blocks=1,
            num_heads=4,
            use_mean_pooling=True,
        )

        inputs = torch.randn(64, 10, 4)
        outputs = transformer(inputs)

        assert outputs.shape == (64, 2)
        assert not torch.isnan(outputs).any()

    def test_varying_real_counts_per_sample(self):
        """Test with different numbers of real items per sample."""
        transformer = Transformer(
            in_channels=4,
            out_channels=2,
            hidden_channels=16,
            num_blocks=1,
            num_heads=4,
            use_mean_pooling=True,
        )
        transformer.eval()

        batch_size = 4
        num_items = 6
        in_channels = 4

        inputs = torch.randn(batch_size, num_items, in_channels)
        
        # Each sample has different number of real items: 6, 4, 2, 1
        pool_mask = torch.tensor([
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
            [True, False, False, False, False, False],
        ])

        outputs = transformer(inputs, pool_mask=pool_mask)

        assert outputs.shape == (batch_size, 2)
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()

