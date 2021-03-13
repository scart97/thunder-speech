from hypothesis import given
from hypothesis.strategies import text

from thunder.text_processing.preprocess import (
    expand_numbers,
    lower_text,
    normalize_text,
)


@given(text())
def test_lower_text(inp):
    lower = lower_text(inp)
    assert lower == inp.lower()


@given(text())
def test_normalize_text(inp):
    normalized = normalize_text(inp)
    assert normalized.isascii()


def test_normalize_text_specific_inputs():
    assert normalize_text("áàâã") == "aaaa"
    assert normalize_text("ç") == "c"


def test_expand_pure_numbers():
    out = expand_numbers("42", "pt_BR")
    assert out == "quarenta e dois"

    out = expand_numbers("42º", "pt_BR")
    assert out == "quadragésimo segundo"


def test_expand_numbers_inside_text():
    out = expand_numbers("abc 42 abc", "pt_BR")
    assert out == "abc quarenta e dois abc"

    out = expand_numbers("abc 43º abc", "pt_BR")
    assert out == "abc quadragésimo terceiro abc"
