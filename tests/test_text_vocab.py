# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from string import ascii_lowercase

import pytest

import torch
from hypothesis import given
from hypothesis.strategies import text

from thunder.text_processing.tokenizer import char_tokenizer
from thunder.text_processing.vocab import Vocab


@pytest.fixture(params=[True, False])
def simple_vocab(request):
    vocab = Vocab(initial_vocab_tokens=[" "] + list(ascii_lowercase))
    if request.param:
        return torch.jit.script(vocab)
    return vocab


def test_vocab_mapping_is_bidirectionally_correct(simple_vocab: Vocab):
    assert len(simple_vocab.itos) == len(simple_vocab.stoi)
    for k, v in simple_vocab.stoi.items():
        assert simple_vocab.itos[v] == k


def test_vocab_blank_is_not_the_unknown(simple_vocab: Vocab):
    assert simple_vocab.blank_idx != simple_vocab.unknown_idx
    assert simple_vocab.blank_token != simple_vocab.unknown_token


def test_numericalize_adds_unknown_token(simple_vocab: Vocab):
    if isinstance(simple_vocab, torch.jit.ScriptModule):
        return
    out = simple_vocab.numericalize(["a", "b", "c", "$"])
    expected = torch.Tensor([1, 2, 3, simple_vocab.unknown_idx])
    assert (out == expected).all()


def test_numericalize_decode_is_bidirectionally_correct(simple_vocab: Vocab):
    if isinstance(simple_vocab, torch.jit.ScriptModule):
        return
    inp = ["a", "b", "c", "d", "e"]
    out1 = simple_vocab.numericalize(inp)
    out = simple_vocab.decode_into_text(out1)
    assert out == inp


def test_add_special_tokens(simple_vocab: Vocab):
    inp = ["a", "b", "c"]
    out = simple_vocab.add_special_tokens(inp)
    assert out == [simple_vocab.start_token, "a", "b", "c", simple_vocab.end_token]


def test_special_idx_are_different(simple_vocab: Vocab):
    all_tokens = set(
        [
            simple_vocab.start_idx,
            simple_vocab.end_idx,
            simple_vocab.pad_idx,
            simple_vocab.unknown_idx,
            simple_vocab.blank_idx,
        ]
    )
    # There's no problem if the blank_idx == pad_idx
    assert len(all_tokens) >= 4


@given(text(min_size=1, max_size=100))
def test_nemo_compat_mode(sample):
    if isinstance(simple_vocab, torch.jit.ScriptModule):
        return
    vocab = Vocab(initial_vocab_tokens=["a", "b", "c"], nemo_compat=True)
    assert len(vocab) == 4
    assert vocab.blank_idx == 3

    out = vocab.numericalize(char_tokenizer(sample))
    if out.numel() > 0:
        assert out.max() < 4
        assert out.min() >= 0
