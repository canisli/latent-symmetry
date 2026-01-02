from functools import partial
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F

# Baseline transformer implementation from
# https://github.com/heidelberg-hepml/lorentz-gatr

def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(
        -1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :]
    )

class BaselineLayerNorm(nn.Module):
    """Baseline layer norm over all dimensions except the first."""

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        outputs : Tensor
            Normalized inputs.
        """
        return torch.nn.functional.layer_norm(
            inputs, normalized_shape=inputs.shape[-1:]
        )


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-head attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_channels, 3 * hidden_channels * num_heads)

    def forward(self, inputs):
        """Forward pass.

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels * num_heads)
        q, k, v = rearrange(
            qkv,
            "... items (qkv hidden_channels num_heads) -> qkv ... num_heads items hidden_channels",
            num_heads=self.num_heads,
            qkv=3,
        )
        return q, k, v


class MultiQueryQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-query attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_linear = nn.Linear(in_channels, hidden_channels)
        self.v_linear = nn.Linear(in_channels, hidden_channels)

    def forward(self, inputs):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        q = rearrange(
            self.q_linear(inputs),
            "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
            num_heads=self.num_heads,
        )
        k = self.k_linear(inputs)[
            ..., None, :, :
        ]  # (..., head=1, item, hidden_channels)
        v = self.v_linear(inputs)[..., None, :, :]
        return q, k, v


class BaselineSelfAttention(nn.Module):
    """Baseline self-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_heads: int = 8,
        multi_query: bool = True,
        dropout_prob=None,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        # Linear maps
        qkv_class = MultiQueryQKVLinear if multi_query else MultiHeadQKVLinear
        self.qkv_linear = qkv_class(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels * num_heads, out_channels)

        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(
            inputs
        )  # each: (..., num_heads, num_items, num_channels, 16)

        # Attention layer
        h = self._attend(q, k, v, attention_mask, is_causal=is_causal)

        # Concatenate heads and transform linearly
        h = rearrange(
            h,
            "... num_heads num_items hidden_channels -> ... num_items (num_heads hidden_channels)",
        )
        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs

    @staticmethod
    def _attend(q, k, v, attention_mask=None, is_causal=False):
        """Scaled dot-product attention."""

        # Add batch dimension if needed
        bh_shape = q.shape[:-2]
        q = to_nd(q, 4)
        k = to_nd(k, 4)
        v = to_nd(v, 4)

        # SDPA
        outputs = scaled_dot_product_attention(
            q.contiguous(),
            k.expand_as(q).contiguous(),
            v.expand_as(q).contiguous(),
            attn_mask=attention_mask,
            is_causal=is_causal,
        )

        # Return batch dimensions to inputs
        outputs = outputs.view(*bh_shape, *outputs.shape[-2:])

        return outputs


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block.

    Inputs are first processed by a block consisting of LayerNorm, multi-head self-attention, and
    residual connection. Then the data is processed by a block consisting of another LayerNorm, an
    item-wise two-layer MLP with GeLU activations, and another residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    num_heads : int
        Number of attention heads.
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        channels,
        num_heads: int = 8,
        increase_hidden_channels=1,
        multi_query: bool = True,
        dropout_prob=None,
    ) -> None:
        super().__init__()

        self.norm = BaselineLayerNorm()

        hidden_channels = channels // num_heads * increase_hidden_channels

        self.attention = BaselineSelfAttention(
            channels,
            channels,
            hidden_channels,
            num_heads=num_heads,
            multi_query=multi_query,
            dropout_prob=dropout_prob,
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
        )

    def forward(
        self, inputs: torch.Tensor, attention_mask=None, is_causal=False
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """

        # Residual attention
        h = self.norm(inputs)
        h = self.attention(h, attention_mask=attention_mask, is_causal=is_causal)
        outputs = inputs + h

        # Residual MLP
        h = self.norm(outputs)
        h = self.mlp(h)
        outputs = outputs + h

        return outputs



class Transformer(nn.Module):
    """Baseline transformer.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    use_mean_pooling : bool
        If True, apply mean pooling over the sequence dimension before the output linear layer.
        If False, return per-item outputs.
    pool_mask : Optional[Tensor]
        Optional mask for mean pooling. If None and use_mean_pooling=True, will auto-detect
        padding from zero inputs. Shape: (..., num_items) where True indicates real items.
    
    Layer Indexing for forward_with_intermediate (with num_blocks=4):
        - layer_idx 0: after linear_in (per-particle)
        - layer_idx 1-4: after each transformer block (per-particle)
        - layer_idx 5: post-pooling (only if use_mean_pooling=True)
        - layer_idx -1: final output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int = 10,
        num_heads: int = 8,
        checkpoint_blocks: bool = False,
        increase_hidden_channels=1,
        multi_query: bool = False,
        dropout_prob=None,
        use_mean_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.use_mean_pooling = use_mean_pooling
        self.num_blocks = num_blocks
        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    increase_hidden_channels=increase_hidden_channels,
                    multi_query=multi_query,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self, inputs: torch.Tensor, attention_mask=None, is_causal=False, pool_mask=None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items, num_channels)
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask for transformer blocks
        is_causal: bool
            Whether to use causal masking
        pool_mask : Optional[Tensor]
            Optional mask for mean pooling. If None and use_mean_pooling=True, will auto-detect
            padding from zero inputs. Shape: (..., num_items) where True indicates real items.

        Returns
        -------
        outputs : Tensor
            If use_mean_pooling=False: shape (..., num_items, out_channels)
            If use_mean_pooling=True: shape (..., out_channels)
        """
        h = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                h = checkpoint(
                    block,
                    inputs=h,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                )
            else:
                h = block(h, attention_mask=attention_mask, is_causal=is_causal)
        

        if self.use_mean_pooling:
            if pool_mask is None:
                pool_mask = torch.any(inputs != 0.0, dim=-1)

            mask = pool_mask.float()
            valid_counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            h = (h * mask.unsqueeze(-1)).sum(dim=-2) / valid_counts
        
        outputs = self.linear_out(h)

        return outputs

    def _apply_pooling(
        self, h: torch.Tensor, inputs: torch.Tensor, pool_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply mean pooling to hidden states."""
        if pool_mask is None:
            pool_mask = torch.any(inputs != 0.0, dim=-1)
        mask = pool_mask.float()
        valid_counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1)).sum(dim=-2) / valid_counts

    def forward_with_intermediate(
        self,
        inputs: torch.Tensor,
        layer_idx: int,
        attention_mask=None,
        is_causal: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning intermediate activations at specified layer.
        
        Parameters
        ----------
        inputs : Tensor
            Input data with shape (..., num_items, in_channels)
        layer_idx : int
            Layer index to return activations from:
            - 0: after linear_in (per-particle)
            - 1..num_blocks: after each transformer block (per-particle)
            - num_blocks + 1: post-pooling (only if use_mean_pooling=True)
            - -1: final output
        attention_mask : None or Tensor
            Optional attention mask
        is_causal : bool
            Whether to use causal masking
        mask : Optional[Tensor]
            Optional mask for pooling, aliased as pool_mask
        
        Returns
        -------
        activations : Tensor
            Intermediate activations at the specified layer
        """
        pool_mask = mask  # Alias for consistency with symmetry loss interface
        
        if layer_idx == -1:
            return self.forward(inputs, attention_mask, is_causal, pool_mask)
        
        pool_layer_idx = self.num_blocks + 1
        
        if layer_idx < 0 or layer_idx > pool_layer_idx:
            raise ValueError(
                f"layer_idx must be between 0 and {pool_layer_idx}, or -1. "
                f"Got {layer_idx}"
            )
        
        if layer_idx == pool_layer_idx and not self.use_mean_pooling:
            raise ValueError(
                f"layer_idx={pool_layer_idx} (post-pooling) is only valid when "
                f"use_mean_pooling=True"
            )
        
        # After linear_in
        h = self.linear_in(inputs)
        if layer_idx == 0:
            return h
        
        # Through transformer blocks
        for i, block in enumerate(self.blocks, start=1):
            if self.checkpoint_blocks:
                h = checkpoint(
                    block,
                    inputs=h,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                )
            else:
                h = block(h, attention_mask=attention_mask, is_causal=is_causal)
            
            if i == layer_idx:
                return h
        
        # Post-pooling (only reachable if use_mean_pooling=True)
        if layer_idx == pool_layer_idx:
            return self._apply_pooling(h, inputs, pool_mask)
        
        # Should not reach here
        raise RuntimeError(f"Failed to return activations for layer_idx={layer_idx}")