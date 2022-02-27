"""
Text preprocessing functionality
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["lower_text", "normalize_text", "expand_numbers"]

import re
import unicodedata

from num2words import num2words


def lower_text(text: str) -> str:
    """Transform all the text to lowercase.

    Args:
        text: Input text

    Returns:
        Output text
    """
    return text.lower()


def normalize_text(text: str) -> str:
    """Normalize the text to remove accents
    and ensure all the characters are valid
    ascii symbols.

    Args:
        text: Input text

    Returns:
        Output text
    """
    nfkd_form = unicodedata.normalize("NFKD", text)
    only_ascii = nfkd_form.encode("ASCII", "ignore")
    return only_ascii.decode()


def expand_numbers(text: str, language: str = "en") -> str:
    """Expand the numbers present inside the text.
    That means converting "42" into "forty two".
    It also detects if the number is ordinal automatically.

    Args:
        text: Input text
        language: Language used to expand the numbers. Defaults to "en".

    Returns:
        Output text
    """
    number_regex = re.compile(r"\d+º*")

    all_numbers = number_regex.findall(text)
    for num in all_numbers:
        if "º" in num:
            pure_number = num.replace("º", "").strip()
            expanded = num2words(int(pure_number), lang=language, to="ordinal")
        else:
            expanded = num2words(int(num), lang=language)
        text = text.replace(num, expanded)
    return text
