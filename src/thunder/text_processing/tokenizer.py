# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from typing import List


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
