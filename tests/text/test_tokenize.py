# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from thunder.text_processing.tokenizer import (
    char_tokenizer,
    get_most_frequent_tokens,
    word_tokenizer,
)


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


def test_get_most_frequent_tokens():
    corpus = "abc abcd abcde abcdef abcdefg abcdefgh"

    output = get_most_frequent_tokens(corpus, char_tokenizer, minimum_frequency=1)
    assert output == ["a", "b", "c", " ", "d", "e", "f", "g", "h"]

    output = get_most_frequent_tokens(corpus, char_tokenizer, minimum_frequency=3)
    assert output == ["a", "b", "c", " ", "d", "e", "f"]

    output = get_most_frequent_tokens(corpus, char_tokenizer, max_number_of_tokens=4)
    assert output == ["a", "b", "c", " "]

    output = get_most_frequent_tokens(
        corpus, char_tokenizer, minimum_frequency=6, max_number_of_tokens=4
    )
    assert output == ["a", "b", "c"]
