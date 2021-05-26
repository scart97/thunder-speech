# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["word_tokenizer", "char_tokenizer"]

from collections import Counter
from typing import Callable, List, Optional


def word_tokenizer(text: str) -> List[str]:
    """Tokenize input text splitting into words

    Args:
        text : Input text

    Returns:
        Tokenized text
    """
    return text.split()


def char_tokenizer(text: str) -> List[str]:
    """Tokenize input text splitting into characters

    Args:
        text : Input text

    Returns:
        Tokenized text
    """
    return list(text)


def get_most_frequent_tokens(
    corpus: str,
    tokenize_function: Callable,
    minimum_frequency: int = 1,
    max_number_of_tokens: Optional[int] = None,
) -> List[str]:
    """Helper function to get the most frequent tokens from a text corpus.

    Args:
        corpus : Text corpus to be used, this is a long string containing all of your text
        tokenize_function : Same tokenizer function that will be used during training
        minimum_frequency : Remove any token with frequency less than that. Defaults to 1.
        max_number_of_tokens : Optionally limit to the K most frequent tokens. Defaults to None.

    Returns:
        All of the unique, most frequent tokens, ordered by frequency.
    """

    tokenized = tokenize_function(corpus)
    token_counter = Counter(tokenized)
    output_tokens = []
    for token, count in token_counter.most_common(max_number_of_tokens):
        if count >= minimum_frequency:
            output_tokens.append(token)
    return output_tokens
