from typing import Any, Callable, List, Optional

import editdistance
from torch import Tensor, tensor
from torchmetrics import Metric


def _edit_update(predicted: str, reference: str):
    distance = editdistance.eval(predicted, reference)
    total = len(reference)
    assert total > 0, "The reference is empty, this will cause a incorrect metric value"
    return distance, total


def _edit_compute(distance, total):
    return float(distance / total)


def _cer_update(predicted: str, reference: str):
    predicted = list(predicted)
    reference = list(reference)
    return _edit_update(predicted, reference)


def cer(predicted: str, reference: str):
    distance, total = _cer_update(predicted, reference)
    return _edit_compute(distance, total)


def _wer_update(predicted: str, reference: str):
    predicted = predicted.split(" ")
    reference = reference.split(" ")
    return _edit_update(predicted, reference)


def wer(predicted: str, reference: str):
    distance, total = _wer_update(predicted, reference)
    return _edit_compute(distance, total)


class _EditBase(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("distance", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: List[str], target: List[str]):
        # fmt: off
        assert len(target) > 0, "You need to pass at least one pair"
        assert len(preds) == len(target), "The number of predictions and targets must be the same"
        # fmt: on

        for predicted, reference in zip(preds, target):
            distance, total = self.update_func(predicted, reference)

            self.distance += distance
            self.total += total

    def update_func(self, predicted, reference):
        pass

    def compute(self) -> Tensor:
        return tensor(_edit_compute(self.distance, self.total))


class CER(_EditBase):
    def update_func(self, predicted: str, reference: str):
        return _cer_update(predicted, reference)


class WER(_EditBase):
    def update_func(self, predicted: str, reference: str):
        return _wer_update(predicted, reference)
