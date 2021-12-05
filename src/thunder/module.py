# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["BaseCTCModule", "load_pretrained"]

from typing import Any, Dict, List, Optional, Tuple, Type

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.text.wer import WER

from thunder.ctc_loss import calculate_ctc
from thunder.registry import load_checkpoint_data
from thunder.text_processing.transform import BatchTextTransformer


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
    ):
        """Base module for all systems that follow the same CTC training procedure.

        Args:
            encoder: Encoder part of the model
            decoder: Decoder part of the model
            audio_transform: Transforms raw audio into the features the encoder expects
            text_transform: Class that encodes and decodes all textual representation
            optimizer_class: Optimizer to use during training.
            optimizer_kwargs: Optional extra kwargs to the optimizer.
            lr_scheduler_class: Optional class to use a learning rate scheduler with the optimizer.
            lr_scheduler_kwargs: Optional extra kwargs to the learning rate scheduler.
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
        self.lr_scheduler_frequency = self.lr_scheduler_kwargs.pop("frequency", "step")

        # Metrics
        self.validation_cer = CharErrorRate()
        self.validation_wer = WER()
        self.example_input_array = (
            torch.randn((10, 16000)),
            torch.randint(100, 16000, (10,)),
        )

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Process the audio tensor to create the predictions.

        Args:
            x : Audio tensor of shape [batch_size, time]

        Returns:
            Tensor with the predictions.
        """
        features, feature_lengths = self.audio_transform(x, lengths)
        encoded, out_lengths = self.encoder(features, feature_lengths)
        return self.decoder(encoded), out_lengths

    @torch.jit.export
    def predict(self, x: Tensor) -> List[str]:
        """Use this function during inference to predict.

        Args:
            x : Audio tensor of shape [batch_size, time]

        Returns:
            A list of strings, each one contains the corresponding transcription to the original batch element.
        """
        audio_lengths = torch.tensor(x.shape[0] * [x.shape[-1]], device=x.device)
        pred, _ = self(x, audio_lengths)
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
        audio, audio_lengths, texts = batch
        y, y_lengths = self.text_transform.encode(texts, device=self.device)

        probabilities, prob_lengths = self(audio, audio_lengths)
        loss = calculate_ctc(
            probabilities,
            y,
            prob_lengths,
            y_lengths,
            self.text_transform.vocab.blank_idx,
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
        audio, audio_lengths, texts = batch
        y, y_lengths = self.text_transform.encode(texts, device=self.device)

        probabilities, prob_lengths = self(audio, audio_lengths)
        loss = calculate_ctc(
            probabilities,
            y,
            prob_lengths,
            y_lengths,
            self.text_transform.vocab.blank_idx,
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "frequency": self.lr_scheduler_frequency,
            },
        }


def load_pretrained(checkpoint_name: str, **load_kwargs) -> BaseCTCModule:
    """Load from the original checkpoint into a LightningModule ready for training or inference.

    Args:
        checkpoint_name : Checkpoint to be downloaded locally and lodaded.
        load_kwargs : Keyword arguments used by the checkpoint loading function.

    Returns:
        The module loaded from the checkpoint
    """
    checkpoint_data = load_checkpoint_data(checkpoint_name, **load_kwargs)
    instantiated = BaseCTCModule(
        checkpoint_data.encoder,
        checkpoint_data.decoder,
        checkpoint_data.audio_transform,
        checkpoint_data.text_transform,
    )
    instantiated.eval()
    return instantiated
