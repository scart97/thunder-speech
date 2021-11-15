from functools import partial
from typing import Callable, Dict, Type

from thunder.citrinet.compatibility import CitrinetCheckpoint, load_citrinet_checkpoint
from thunder.quartznet.compatibility import (
    QuartznetCheckpoint,
    load_quartznet_checkpoint,
)
from thunder.utils import BaseCheckpoint, CheckpointResult

CHECKPOINT_LOAD_FUNC_TYPE = Callable[..., CheckpointResult]

CHECKPOINT_REGISTRY: Dict[str, CHECKPOINT_LOAD_FUNC_TYPE] = {}


def register_checkpoint(
    checkpoints: Type[BaseCheckpoint], load_function: CHECKPOINT_LOAD_FUNC_TYPE
):
    for checkpoint in checkpoints:
        CHECKPOINT_REGISTRY.update(
            {checkpoint.name: partial(load_function, checkpoint)}
        )


register_checkpoint(QuartznetCheckpoint, load_quartznet_checkpoint)
register_checkpoint(CitrinetCheckpoint, load_citrinet_checkpoint)
