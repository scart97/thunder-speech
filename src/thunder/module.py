"""
Base module to train ctc models
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["BaseCTCModule"]

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.text.wer import WordErrorRate

from thunder.ctc_loss import calculate_ctc
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
        encoder_final_dimension: int = None,
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
            encoder_final_dimension: number of features in the encoder output.
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
        self.lr_scheduler_interval = self.lr_scheduler_kwargs.pop("interval", "step")

        self.encoder_final_dimension = encoder_final_dimension

        # Metrics
        self.validation_cer = CharErrorRate()
        self.validation_wer = WordErrorRate()
        self.example_input_array = (
            torch.randn((10, 16000)),
            torch.randint(100, 16000, (10,)),
        )

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Process the audio tensor to create the predictions.

        Args:
            x: Audio tensor of shape [batch_size, time]

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
            x: Audio tensor of shape [batch_size, time]

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
            batch: Tuple containing the batched audios, normalized lengths and the corresponding text labels.
            batch_idx: Batch index

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
            batch: Tuple containing the batched audios, normalized lengths and the corresponding text labels.
            batch_idx: Batch index

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

    def estimated_steps_per_epoch(self) -> int:
        """Training steps per epoch inferred from datamodule and devices.

        Modified directly from lightning-flash:
        https://github.com/PyTorchLightning/lightning-flash/blob/e73c420d66cb531892fd9032a8328b81e15e0d62/flash/core/model.py#L984
        """
        if not getattr(self, "trainer", None):
            raise MisconfigurationException(
                "The LightningModule isn't attached to the trainer yet."
            )
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        return dataset_size // effective_batch_size

    def estimated_max_steps(self) -> int:
        """Estimate max number of steps during the full training.

        Modified directly from lightning-flash:
        https://github.com/PyTorchLightning/lightning-flash/blob/e73c420d66cb531892fd9032a8328b81e15e0d62/flash/core/model.py#L1002
        """
        estimated_max_steps = self.estimated_steps_per_epoch() * self.trainer.max_epochs
        if self.trainer.max_steps and self.trainer.max_steps < estimated_max_steps:
            return self.trainer.max_steps
        return estimated_max_steps

    def _update_special_optimizer_args(self, original_kwargs: Dict) -> Dict:
        updated_kwargs = original_kwargs.copy()

        epochs_arg = updated_kwargs.pop("epochs_arg", None)
        if epochs_arg:
            updated_kwargs[epochs_arg] = self.trainer.max_epochs

        steps_arg = updated_kwargs.pop("steps_arg", None)
        if steps_arg:
            updated_kwargs[steps_arg] = self.estimated_steps_per_epoch()

        total_steps_arg = updated_kwargs.pop("total_steps_arg", None)
        if total_steps_arg:
            updated_kwargs[total_steps_arg] = self.estimated_max_steps()
        return updated_kwargs

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optim_kwargs = self._update_special_optimizer_args(self.optimizer_kwargs)
        optimizer = self.optimizer_class(
            filter(lambda p: p.requires_grad, self.parameters()), **optim_kwargs
        )
        if not self.lr_scheduler_class:
            return optimizer

        scheduler_kwargs = self._update_special_optimizer_args(self.lr_scheduler_kwargs)
        lr_scheduler = self.lr_scheduler_class(optimizer, **scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.lr_scheduler_interval,
            },
        }
