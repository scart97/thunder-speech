# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from string import ascii_lowercase

import pytest

import torch
from hypothesis import given
from hypothesis.strategies import text

from thunder.text_processing.tokenizer import char_tokenizer
from thunder.text_processing.vocab import SimpleVocab, Vocab


@pytest.fixture(params=[True, False])
def complex_vocab(request):
    vocab = Vocab(initial_vocab_tokens=[" "] + list(ascii_lowercase))
    if request.param:
        return torch.jit.script(vocab)
    return vocab


@pytest.fixture(params=[True, False])
def simple_vocab(request):
    vocab = SimpleVocab(initial_vocab_tokens=[" "] + list(ascii_lowercase))
    if request.param:
        return torch.jit.script(vocab)
    return vocab


def test_vocab_mapping_is_bidirectionally_correct(complex_vocab: Vocab):
    assert len(complex_vocab.itos) == len(complex_vocab.stoi)
    for k, v in complex_vocab.stoi.items():
        assert complex_vocab.itos[v] == k


def test_nemo_vocab_mapping_is_bidirectionally_correct(simple_vocab: SimpleVocab):
    assert len(simple_vocab.itos) == len(simple_vocab.stoi)
    for k, v in simple_vocab.stoi.items():
        assert simple_vocab.itos[v] == k


def test_vocab_blank_is_not_the_unknown(complex_vocab: Vocab):
    assert complex_vocab.blank_idx != complex_vocab.unknown_idx
    assert complex_vocab.blank_token != complex_vocab.unknown_token


def test_numericalize_adds_unknown_token(complex_vocab: Vocab):
    if isinstance(complex_vocab, torch.jit.ScriptModule):
        return
    out = complex_vocab.numericalize(["a", "b", "c", "$"])
    expected = torch.Tensor([1, 2, 3, complex_vocab.unknown_idx])
    assert (out == expected).all()


def test_numericalize_nemo_ignores_unknown(simple_vocab: SimpleVocab):
    if isinstance(simple_vocab, torch.jit.ScriptModule):
        return
    out = simple_vocab.numericalize(["a", "b", "c", "$"])
    expected = torch.Tensor([1, 2, 3])
    assert (out == expected).all()


def test_numericalize_decode_is_bidirectionally_correct(complex_vocab: Vocab):
    if isinstance(complex_vocab, torch.jit.ScriptModule):
        return
    inp = ["a", "b", "c", "d", "e"]
    out1 = complex_vocab.numericalize(inp)
    out = complex_vocab.decode_into_text(out1)
    assert out == inp


def test_nemo_numericalize_decode_is_bidirectionally_correct(simple_vocab: SimpleVocab):
    if isinstance(simple_vocab, torch.jit.ScriptModule):
        return
    inp = ["a", "b", "c", "d", "e"]
    out1 = simple_vocab.numericalize(inp)
    out = simple_vocab.decode_into_text(out1)
    assert out == inp


def test_add_special_tokens(complex_vocab: Vocab):
    inp = ["a", "b", "c"]
    out = complex_vocab.add_special_tokens(inp)
    assert out == [complex_vocab.start_token, "a", "b", "c", complex_vocab.end_token]


def test_nemo_doesnt_add_special_tokens(simple_vocab: SimpleVocab):
    inp = ["a", "b", "c"]
    out = simple_vocab.add_special_tokens(inp)
    assert out == ["a", "b", "c"]


def test_special_idx_are_different(complex_vocab: Vocab):
    all_tokens = set(
        [
            complex_vocab.start_idx,
            complex_vocab.end_idx,
            complex_vocab.pad_idx,
            complex_vocab.unknown_idx,
            complex_vocab.blank_idx,
        ]
    )
    # There's no problem if the blank_idx == pad_idx
    assert len(all_tokens) >= 4


@given(text(min_size=1, max_size=100))
def test_nemo_compat_mode(sample):
    vocab = SimpleVocab(initial_vocab_tokens=["a", "b", "c"])
    assert len(vocab) == 4
    assert vocab.blank_idx == 3

    out = vocab.numericalize(char_tokenizer(sample))
    if out.numel() > 0:
        assert out.max() < 4
        assert out.min() >= 0
