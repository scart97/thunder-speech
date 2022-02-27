"""Functionality to register the multiple checkpoints and provide a unified loading interface.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["register_checkpoint_enum", "load_pretrained"]

from functools import partial
from typing import Callable, Dict, Type, Union

from thunder.citrinet.compatibility import CitrinetCheckpoint, load_citrinet_checkpoint
from thunder.huggingface.compatibility import load_huggingface_checkpoint
from thunder.module import BaseCTCModule
from thunder.quartznet.compatibility import (
    QuartznetCheckpoint,
    load_quartznet_checkpoint,
)
from thunder.utils import BaseCheckpoint

CHECKPOINT_LOAD_FUNC_TYPE = Callable[..., BaseCTCModule]

CHECKPOINT_REGISTRY: Dict[str, CHECKPOINT_LOAD_FUNC_TYPE] = {}


def register_checkpoint_enum(
    checkpoints: Type[BaseCheckpoint], load_function: CHECKPOINT_LOAD_FUNC_TYPE
):
    """Register all variations of some checkpoint enum with the corresponding loading function

    Args:
        checkpoints: Base checkpoint class
        load_function: function to load the checkpoint,
            must receive one instance of `checkpoints` as first argument"""
    for checkpoint in checkpoints:
        CHECKPOINT_REGISTRY.update(
            {checkpoint.name: partial(load_function, checkpoint)}
        )


register_checkpoint_enum(QuartznetCheckpoint, load_quartznet_checkpoint)
register_checkpoint_enum(CitrinetCheckpoint, load_citrinet_checkpoint)


def load_pretrained(
    checkpoint_name: Union[str, BaseCheckpoint], **load_kwargs
) -> BaseCTCModule:
    """Load data from any registered checkpoint

    Args:
        checkpoint_name: Base checkpoint name, like "QuartzNet5x5LS_En" or "facebook/wav2vec2-large-960h"

    Returns:
        Object containing the checkpoint data (encoder, decoder, transforms and additional data).
    """
    if isinstance(checkpoint_name, BaseCheckpoint):
        checkpoint_name = checkpoint_name.name
    # Special case when dealing with any huggingface model
    if "/" in checkpoint_name:
        model_data = load_huggingface_checkpoint(checkpoint_name, **load_kwargs)
    else:
        load_fn = CHECKPOINT_REGISTRY[checkpoint_name]
        model_data = load_fn(**load_kwargs)
    return model_data
