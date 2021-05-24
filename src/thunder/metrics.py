__all__ = ["single_cer", "single_wer", "CER", "WER", "EditBaseMetric"]

from typing import Any, Callable, List, Optional, Tuple

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


def single_cer(predicted: str, reference: str) -> float:
    """Computes the character error rate between one prediction and the corresponding reference.
    This is the functional form, and it's recommended to use inside the training loop the class
    [`CER`][thunder.metrics.CER] instead.

    Args:
        predicted : Model prediction after decoding back to string
        reference : Reference text

    Returns:
        Value between 0.0 and 1.0 that measures the error rate.
    """
    distance, total = _cer_update(predicted, reference)
    return _edit_compute(distance, total)


def _wer_update(predicted: str, reference: str):
    predicted = predicted.split(" ")
    reference = reference.split(" ")
    return _edit_update(predicted, reference)


def single_wer(predicted: str, reference: str) -> float:
    """Computes the word error rate between one prediction and the corresponding reference.
    This is the functional form, and it's recommended to use inside the training loop the class
    [`WER`][thunder.metrics.WER] instead.

    Args:
        predicted : Model prediction after decoding back to string
        reference : Reference text

    Returns:
        Value between 0.0 and 1.0 that measures the error rate.
    """

    distance, total = _wer_update(predicted, reference)
    return _edit_compute(distance, total)


class EditBaseMetric(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        """Base metric for computations based on edit distance.

        Args:
            compute_on_step:
                Forward only calls update() and returns None if this is set to False.
            dist_sync_on_step:
                Synchronize metric state across processes at each forward() before returning the value at the step.
            process_group:
                Specify the process group on which synchronization is called. default: None (which selects the entire world)
            dist_sync_fn:
                Callback that performs the allgather operation on the metric state. When None, DDP will be used to perform the allgather.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("distance", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: List[str], target: List[str]):
        """Method used to update the internal counters of the metric at every batch.
        Subclasses should leave this function untouched and implement
        update_func instead.

        Args:
            preds : List of predictions of the model, already decoded into string form.
            target : List of corresponding references.
        """
        # fmt: off
        assert len(target) > 0, "You need to pass at least one pair"
        assert len(preds) == len(target), "The number of predictions and targets must be the same"
        # fmt: on

        for predicted, reference in zip(preds, target):
            distance, total = self.update_func(predicted, reference)

            self.distance += distance
            self.total += total

    def update_func(self, predicted: str, reference: str) -> Tuple[int, int]:
        """Function to calculate the statistics from one pair of elements.
        This function should take the two strings, split them according to
        characters/words/phonemes based on the metric implemented, calculate
        the edit distance between the two splitted strings and return also the
        normalizing factor, that is the number of elements in the splitted
        reference.

        Args:
            predicted : Single model prediction
            reference : Corresponding reference

        Returns:
            Tuple containing the pure edit distance between predicted and reference, and the normalizing factor.
        """
        pass

    def compute(self) -> Tensor:
        """Uses the aggregated counters to calculate the final metric value.

        Returns:
            Float tensor between 0.0 and 1.0 representing the error rate.
        """
        return tensor(_edit_compute(self.distance, self.total))


class CER(EditBaseMetric):
    """Metric to compute the character error rate of predictions during the training loop.
    Accepts lists of predictions and references, correctly accumulating the metrics and
    computing the final value when requested.
    Check [`EditBaseMetric`][thunder.metrics.EditBaseMetric] for more details on the
    possible methods.
    """

    def update_func(self, predicted: str, reference: str) -> Tuple[int, int]:
        """Compute the statistics used to compare two strings using
        character error rate.

        Args:
            predicted : Single model prediction
            reference : Corresponding reference

        Returns:
            Tuple containing the pure edit distance between predicted and reference, and the normalizing factor.
        """
        return _cer_update(predicted, reference)


class WER(EditBaseMetric):
    """Metric to compute the word error rate of predictions during the training loop.
    Accepts lists of predictions and references, correctly accumulating the metrics and
    computing the final value when requested.
    Check [`EditBaseMetric`][thunder.metrics.EditBaseMetric] for more details on the
    possible methods.
    """

    def update_func(self, predicted: str, reference: str) -> Tuple[int, int]:
        """Compute the statistics used to compare two strings using
        word error rate.

        Args:
            predicted : Single model prediction
            reference : Corresponding reference

        Returns:
            Tuple containing the pure edit distance between predicted and reference, and the normalizing factor.
        """
        return _wer_update(predicted, reference)
