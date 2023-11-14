# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from string import ascii_lowercase
from typing import List

import pytest

import torch

from thunder.text_processing.tokenizer import word_tokenizer
from thunder.text_processing.transform import BatchTextTransformer


@pytest.fixture(params=[True, False])
def tfm(request):
    transform = BatchTextTransformer(
        tokens=[" "] + list(ascii_lowercase),
        blank_token="<blank>",
        pad_token="<blank>",
        unknown_token="<unk>",
        start_token="<bos>",
        end_token="<eos>",
    )
    if request.param:
        return torch.jit.script(transform)
    return transform


@pytest.fixture
def blank_input(tfm):
    # A tensor full of blank probabilities
    # torchscript seems to nuke the properties, cant use tfm.num_tokens
    x = torch.zeros(1, len(tfm.vocab.itos), 100)
    x[:, tfm.vocab.blank_idx, :] = 1
    return x


def test_encode_text(tfm: BatchTextTransformer):
    # Skip this test if the module is exported for inference
    # as it only contains the decoding part.
    if isinstance(tfm, torch.jit.ScriptModule):
        return

    encoded, encoded_lens = tfm.encode(["hello world", "oi"], return_length=True)
    assert len(encoded) == 2
    assert len(encoded_lens) == 2
    expected = torch.Tensor([29, 8, 5, 12, 12, 15, 0, 23, 15, 18, 12, 4, 30])
    assert (encoded[0] == expected).all()
    assert encoded_lens[0] == 13
    assert encoded_lens[1] == 4

    encoded2 = tfm.encode(["hello world", "oi"], return_length=False)
    assert (encoded == encoded2).all()


def test_decoder_remove_blanks(tfm: BatchTextTransformer, blank_input):
    out = tfm.decode_prediction(blank_input.argmax(1))

    assert len(out) == 1
    assert isinstance(out, list)
    assert isinstance(out[0], str)
    assert out[0] == ""


def test_decoder_simple_sequence(tfm: BatchTextTransformer, blank_input):
    a_idx = tfm.vocab.stoi["a"]
    b_idx = tfm.vocab.stoi["b"]
    blank_input[:, a_idx, :10] = 2  # a
    blank_input[:, b_idx, 15:20] = 2  # b

    out = tfm.decode_prediction(blank_input.argmax(1))

    assert len(out) == 1
    assert isinstance(out, list)
    assert isinstance(out[0], str)
    assert out[0] == "ab"


def test_decoder_repeat_same_element(tfm: BatchTextTransformer, blank_input):
    a_idx = tfm.vocab.stoi["a"]
    blank_input[:, a_idx, :10] = 2  # a
    blank_input[:, a_idx, 15:20] = 2  # a
    out = tfm.decode_prediction(blank_input.argmax(1))

    assert len(out) == 1
    assert isinstance(out, list)
    assert isinstance(out[0], str)
    assert out[0] == "aa"


def test_custom_tokenizer():
    def _tok_func(s: str) -> List[str]:
        return s.split(".")

    transform = BatchTextTransformer(
        tokens=[" "] + list(ascii_lowercase),
        blank_token="<blank>",
        custom_tokenizer_function=_tok_func,
    )
    encoded, encoded_lens = transform.encode(
        ["h.e.l.l.o. .w.o.r.l.d", "o.i"], return_length=True
    )
    assert len(encoded) == 2
    assert len(encoded_lens) == 2
    expected = torch.Tensor([8, 5, 12, 12, 15, 0, 23, 15, 18, 12, 4])

    assert (encoded[0] == expected).all()
    assert encoded_lens[0] == 11
    assert encoded_lens[1] == 2


def test_word_tokenizer():
    transform = BatchTextTransformer(
        tokens=["hello", "world", "oi"],
        blank_token="<blank>",
        custom_tokenizer_function=word_tokenizer,
    )
    encoded, encoded_lens = transform.encode(["hello world", "oi"], return_length=True)
    assert len(encoded) == 2
    assert len(encoded_lens) == 2
    expected = torch.Tensor([0, 1])

    assert (encoded[0] == expected).all()
    assert encoded_lens[0] == 2
    assert encoded_lens[1] == 1
