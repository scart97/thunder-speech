from typing import List


class Token(str):
    """Tokens are just semantically special strings, but share all the functionality."""

    pass


def word_tokenizer(text: str) -> List[Token]:
    """Tokenize input text splitting into words

    Args:
        text : Input text

    Returns:
        Tokenized text
    """
    return text.split()


def char_tokenizer(text: str) -> List[Token]:
    """Tokenize input text splitting into characters

    Args:
        text : Input text

    Returns:
        Tokenized text
    """
    return list(text)
