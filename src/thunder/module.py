# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["BaseCTCModule", "load_pretrained"]

from typing import Any, Dict, List, Tuple, Type

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from thunder.ctc_loss import calculate_ctc
from thunder.metrics import CER, WER
from thunder.registry import CHECKPOINT_REGISTRY
from thunder.text_processing.transform import BatchTextTransformer
from thunder.wav2vec.compatibility import load_huggingface_checkpoint


class BaseCTCModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        audio_transform: nn.Module,
        text_transform: BatchTextTransformer,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Dict = None,
        lr_scheduler_class: Any = None,
        lr_scheduler_kwargs: Dict = None,
        example_input_array: Tensor = torch.randn((10, 16000)),
    ):
        """Base module for all systems that follow the same CTC training procedure.

        Args:
            encoder: Encoder part of the model
            decoder: Decoder part of the model
            audio_transform: Transforms raw audio into the features the encoder expects
            text_transform: Class that encodes and decodes all textual representation
            optimizer_class: Optimizer to use during training. Defaults to torch.optim.AdamW.
            optimizer_kwargs: Optional extra kwargs to the optimizer. Defaults to None.
            lr_scheduler_class: Optional class to use a learning rate scheduler with the optimizer. Defaults to None.
            lr_scheduler_kwargs: Optional extra kwargs to the learning rate scheduler. Defaults to None.
            example_input_array: Example input, use by pytorch lightning to print nice stuff before training. Defaults to torch.randn((10, 16000)).
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.audio_transform = audio_transform
        self.text_transform = text_transform

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Metrics
        self.validation_cer = CER()
        self.validation_wer = WER()
        self.example_input_array = example_input_array

    def forward(self, x: Tensor) -> Tensor:
        """Process the audio tensor to create the predictions.

        Args:
            x : Audio tensor of shape [batch_size, time]

        Returns:
            Tensor with the predictions.
        """
        features = self.audio_transform(x)
        encoded = self.encoder(features)
        return self.decoder(encoded)

    @torch.jit.export
    def predict(self, x: Tensor) -> List[str]:
        """Use this function during inference to predict.

        Args:
            x : Audio tensor of shape [batch_size, time]

        Returns:
            A list of strings, each one contains the corresponding transcription to the original batch element.
        """
        pred = self(x)
        return self.text_transform.decode_prediction(pred.argmax(1))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        """Training step. Check the original lightning docs for more information.

        Args:
            batch : Tuple containing the batched audios, normalized lengths and the corresponding text labels.
            batch_idx : Batch index

        Returns:
            Training loss for that batch
        """
        audio, audio_lens, texts = batch
        y, y_lens = self.text_transform.encode(texts, device=self.device)

        probabilities = self(audio)
        loss = calculate_ctc(
            probabilities, y, audio_lens, y_lens, self.text_transform.vocab.blank_idx
        )

        self.log("loss/train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        """Validation step. Check the original lightning docs for more information.

        Args:
            batch : Tuple containing the batched audios, normalized lengths and the corresponding text labels.
            batch_idx : Batch index

        Returns:
            Validation loss for that batch
        """
        audio, audio_lens, texts = batch
        y, y_lens = self.text_transform.encode(texts, device=self.device)

        probabilities = self(audio)
        loss = calculate_ctc(
            probabilities, y, audio_lens, y_lens, self.text_transform.vocab.blank_idx
        )

        decoded_preds = self.text_transform.decode_prediction(probabilities.argmax(1))
        decoded_targets = self.text_transform.decode_prediction(
            y, remove_repeated=False
        )
        self.validation_cer(decoded_preds, decoded_targets)
        self.validation_wer(decoded_preds, decoded_targets)

        self.log("loss/val_loss", loss)
        self.log("metrics/cer", self.validation_cer, on_epoch=True)
        self.log("metrics/wer", self.validation_wer, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.optimizer_kwargs
        )
        if not self.lr_scheduler_class:
            return optimizer

        lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
        return [optimizer], [lr_scheduler]


def load_pretrained(checkpoint: str, **load_kwargs) -> BaseCTCModule:
    """Load from the original checkpoint into a LightningModule ready for training or inference.

    Args:
        checkpoint : Checkpoint to be downloaded locally and lodaded.
        load_kwargs : Keyword arguments used by the checkpoint loading function.

    Returns:
        The module loaded from the checkpoint
    """
    # Special case when dealing with any huggingface model
    if "/" in checkpoint:
        encoder_data = load_huggingface_checkpoint(checkpoint, **load_kwargs)
    else:
        load_fn = CHECKPOINT_REGISTRY[checkpoint]
        encoder_data = load_fn(**load_kwargs)
    instantiated = BaseCTCModule(
        encoder_data.encoder,
        encoder_data.decoder,
        encoder_data.audio_transform,
        encoder_data.text_transform,
    )
    instantiated.eval()
    return instantiated
