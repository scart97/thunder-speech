from string import ascii_lowercase

import pytest

import torch

from thunder.text_processing.transform import BatchTextTransformer
from thunder.text_processing.vocab import Vocab


@pytest.fixture
def tfm():
    vocab = Vocab([" "] + list(ascii_lowercase))
    return BatchTextTransformer(vocab=vocab)


def test_encode_text(tfm: BatchTextTransformer):
    encoded, encoded_lens = tfm.encode(["Hello world", "Oi"], return_length=True)
    assert len(encoded) == 2
    assert len(encoded_lens) == 2
    expected = torch.Tensor([2, 12, 9, 16, 16, 19, 4, 27, 19, 22, 16, 8, 3])
    assert (encoded[0] == expected).all()
    assert encoded_lens[0] == 13
    assert encoded_lens[1] == 4

    encoded2 = tfm.encode(["Hello world", "Oi"], return_length=False)
    assert (encoded == encoded2).all()
