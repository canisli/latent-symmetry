import torch

from models import Transformer


def _make_identity_transformer():
    """Transformer with deterministic linear layers for easy verification."""
    model = Transformer(
        in_channels=3,
        out_channels=4,
        hidden_channels=4,
        num_blocks=0,
        use_mean_pooling=True,
    )

    with torch.no_grad():
        model.linear_in.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
        )
        model.linear_in.bias.zero_()
        model.linear_out.weight.copy_(torch.eye(4))
        model.linear_out.bias.zero_()

    return model


def test_mean_pooling_matches_manual_masking():
    model = _make_identity_transformer()

    inputs = torch.tensor(
        [
            # two real tokens, two padded
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            # two real tokens (1st and 3rd), others padded
            [
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )
    pool_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )

    # Manual masked mean on the pre-output hidden states
    hidden = torch.matmul(inputs, model.linear_in.weight.t())  # (batch, items, hidden)
    masked_hidden = hidden * pool_mask.unsqueeze(-1).float()
    expected = masked_hidden.sum(dim=1) / pool_mask.sum(dim=1, keepdim=True).float()

    output = model(inputs, pool_mask=pool_mask)
    assert torch.allclose(output, expected, atol=1e-6)


def test_mean_pooling_auto_mask_matches_explicit():
    model = _make_identity_transformer()

    inputs = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    explicit_mask = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)

    output_auto = model(inputs)  # pool_mask inferred from zeros
    output_explicit = model(inputs, pool_mask=explicit_mask)

    assert torch.allclose(output_auto, output_explicit, atol=1e-6)

