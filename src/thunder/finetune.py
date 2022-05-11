""" Module that implements easy finetuning of any model in the library.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

from typing import Any, Dict, List

import torch

from thunder.module import BaseCTCModule
from thunder.registry import load_pretrained
from thunder.text_processing.transform import BatchTextTransformer
from thunder.utils import ModuleBuilderType, OptimizerBuilderType, SchedulerBuilderType


class FinetuneCTCModule(BaseCTCModule):
    def __init__(
        self,
        checkpoint_name: str,
        checkpoint_kwargs: Dict[str, Any] = None,
        decoder_class: ModuleBuilderType = None,
        decoder_kwargs: Dict[str, Any] = None,
        tokens: List[str] = None,
        text_kwargs: Dict[str, Any] = None,
        optimizer_class: OptimizerBuilderType = torch.optim.AdamW,
        optimizer_kwargs: Dict[str, Any] = None,
        lr_scheduler_class: SchedulerBuilderType = None,
        lr_scheduler_kwargs: Dict[str, Any] = None,
    ):
        """Generic finetune module, load any combination of encoder/decoder and custom tokens

        Args:
            checkpoint_name: Name of the base checkpoint to load
            checkpoint_kwargs: Additional kwargs to the checkpoint loading function.
            decoder_class: Optional class to override the loaded checkpoint.
            decoder_kwargs: Additional kwargs to the decoder_class.
            tokens: If passed a list of tokens, the decoder from the base checkpoint will be replaced by the one in decoder_class, and a new text transform will be build using those tokens.
            text_kwargs: Additional kwargs to the text_tranform class, when tokens is not None.
            optimizer_class: Optimizer to use during training.
            optimizer_kwargs: Optional extra kwargs to the optimizer.
            lr_scheduler_class: Optional class to use a learning rate scheduler with the optimizer.
            lr_scheduler_kwargs: Optional extra kwargs to the learning rate scheduler.
        """
        self.save_hyperparameters()
        checkpoint_kwargs = checkpoint_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        text_kwargs = text_kwargs or {}

        if tokens is not None and decoder_class is None:
            # Missing decoder
            raise ValueError(
                "New tokens were specified, but the module also needs to know the decoder class to initialize properly."
            )

        if tokens is None and decoder_class is not None:
            # Missing tokens
            raise ValueError(
                "A new decoder was specified, but the module also needs to know the tokens to initialize properly."
            )

        checkpoint_data = load_pretrained(checkpoint_name, **checkpoint_kwargs)

        if decoder_class is None:
            # Keep original decoder/text processing
            text_transform = checkpoint_data.text_transform
            decoder = checkpoint_data.decoder
        else:
            # Changing the decoder layer and text processing
            text_transform = BatchTextTransformer(tokens, **text_kwargs)
            decoder = decoder_class(
                checkpoint_data.encoder_final_dimension,
                text_transform.num_tokens,
                **decoder_kwargs,
            )

        super().__init__(
            encoder=checkpoint_data.encoder,
            decoder=decoder,
            audio_transform=checkpoint_data.audio_transform,
            text_transform=text_transform,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
