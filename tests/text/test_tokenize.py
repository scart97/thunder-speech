# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path
from string import ascii_letters
from tempfile import TemporaryDirectory

import pytest

from hypothesis import given
from hypothesis.strategies import text

from thunder.text_processing.tokenizer import (
    BPETokenizer,
    SentencepieceModelFile,
    char_tokenizer,
    get_most_frequent_tokens,
    train_sentencepiece_model,
    word_tokenizer,
)


def test_bpe_tokenizer_load():
    tokenizer = BPETokenizer("tests/nemo_config_samples/example_tokenizer.model")
    assert tokenizer is not None

    with pytest.raises(OSError, match="Not found"):
        BPETokenizer("incorrect_name.model")
    with pytest.raises(OSError, match="Not found"):
        BPETokenizer("")


def test_bpe_tokenizer():
    tokenizer = BPETokenizer("tests/nemo_config_samples/example_tokenizer.model")

    out = tokenizer("Hello world")
    assert out == ["▁", "H", "el", "lo", "▁world"]

    out = tokenizer("hello world")
    assert out == ["▁he", "ll", "o", "▁world"]


def test_bpe_tokenizer_empty_input():
    tokenizer = BPETokenizer("tests/nemo_config_samples/example_tokenizer.model")
    assert tokenizer("") == []


@given(text(list(ascii_letters) + ["$", "@", " ", "#", "!"]))
def test_bpe_tokenizer_reversible(text):
    tokenizer = BPETokenizer("tests/nemo_config_samples/example_tokenizer.model")
    out = tokenizer(text)
    back_to_original = "".join(out).replace("▁", " ")
    # Sentencepiece ignores multiple spaces in sequence
    assert back_to_original.strip() == " ".join(text.split())


def test_train_sentencepiece():
    file_with_text = "tests/nemo_config_samples/QuartzNet5x5LS-En.yaml"
    with TemporaryDirectory() as output_dir:
        trained = train_sentencepiece_model(
            file_with_text,
            vocab_size=50,
            output_dir=str(output_dir),
            do_lower_case=True,
            sample_size=150,
        )
        assert isinstance(trained, SentencepieceModelFile)
        assert Path(trained.model_path).exists()
        assert len(trained.vocabulary_tokens) <= 50

        with pytest.warns(UserWarning):
            # Trained model already exists, emit warning
            train_skipped = train_sentencepiece_model(
                file_with_text, vocab_size=50, output_dir=str(output_dir)
            )
            assert train_skipped.model_path == trained.model_path
            assert train_skipped.vocabulary_tokens == trained.vocabulary_tokens


def test_train_sentencepiece_exception():
    with TemporaryDirectory() as output_dir:
        with pytest.raises(ValueError):
            train_sentencepiece_model(
                f"{output_dir}/this_doesnt_exist.txt",
                vocab_size=50,
                output_dir=str(output_dir),
                do_lower_case=True,
                sample_size=150,
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
