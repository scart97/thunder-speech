import torch

from thunder.blocks import lengths_to_mask


def test_lengths_to_mask():
    lengths = torch.tensor([8, 5, 4, 3, 1])
    mask = lengths_to_mask(lengths, max_length=8)
    expected = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
        ]
    )
    assert torch.allclose(mask, expected)
