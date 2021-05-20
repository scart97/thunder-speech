# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from thunder.text_processing.tokenizer import char_tokenizer, word_tokenizer


def test_word_tokenizer():
    out = word_tokenizer("Hello world 123")
    assert out == ["Hello", "world", "123"]


def test_word_tokenizer_empty_input():
    assert word_tokenizer("") == []
    assert word_tokenizer("     ") == []


def test_char_tokenizer():
    out = char_tokenizer("Hello world 123")
    # fmt: off
    assert out == ["H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", " ", "1", "2", "3"]
    # fmt: on


def test_char_tokenizer_empty_input():
    assert char_tokenizer("") == []
    assert char_tokenizer("     ") == [" ", " ", " ", " ", " "]
