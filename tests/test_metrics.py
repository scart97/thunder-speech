import pytest

from hypothesis import given
from hypothesis.strategies import lists, text

from thunder.metrics import CER, WER, single_cer, single_wer


@given(lists(text(min_size=1), min_size=1, max_size=32))
@pytest.mark.parametrize("metric_class", [CER, WER])
def test_metric_class_zero_error(metric_class, test_elements):
    metric = metric_class()
    metric(test_elements, test_elements)
    assert metric.compute() == 0.0


@pytest.mark.parametrize("metric_class", [CER, WER])
def test_geral_metric_error(metric_class):
    metric = metric_class()
    with pytest.raises(AssertionError):
        metric([], [])


def test_cer_specific_error():
    with pytest.raises(AssertionError):
        single_cer("", "")


def test_cer_known_cases():
    # 1 change (remove b) with label_len == 2
    assert single_cer("a", "ab") == 1 / 2

    # 1 change (add c)
    assert single_cer("abc", "ab") == 1 / 2

    # 1 change (c -> b)
    assert single_cer("ac", "ab") == 1 / 2

    # 2 removes
    assert single_cer("", "ab") == 2 / 2

    # 2 adds
    assert single_cer("abcd", "ab") == 2 / 2

    # 2 replaces
    assert single_cer("cd", "ab") == 2 / 2

    # 1 remove with space
    assert single_cer("a ", "a b") == 1 / 3

    # 1 add with space
    assert single_cer("a bc", "a b") == 1 / 3

    # 1 replace with space
    assert single_cer("a c", "a b") == 1 / 3

    # 4 changes (add z, remove f, a->p, d->o)
    assert single_cer("p bzco eg", "a bcd efg") == 4 / 9


def test_wer_known_cases():
    # 1 word change (remove b) with label_len == 1
    assert single_wer("a ", "a b") == 1 / 2

    # 1 word change (add c)
    assert single_wer("a b c", "a b") == 1 / 2

    # 1 word change (c -> b)
    assert single_wer("a c", "a b") == 1 / 2

    # 2 removes
    assert single_wer("", "a b") == 2 / 2

    # 2 adds
    assert single_wer("a b c d", "a b") == 2 / 2

    # 2 replaces
    assert single_wer("c d", "a b") == 2 / 2

    # 4 changes (add z, remove f, a->p, d->o)
    assert single_wer("p b c o e g z", "a b c d e f g") == 4 / 7


def test_multiple_calls_to_metric_class():
    metric = WER()
    reference = ["lalala", "a b c", "x y z"]
    prediction = ["lalala", "a c", "x y z w"]
    metric(prediction, reference)
    metric(prediction, reference)
    metric(prediction, reference)
    metric(prediction, reference)
    metric(prediction, reference)
    assert metric.compute() == 2 / 7
