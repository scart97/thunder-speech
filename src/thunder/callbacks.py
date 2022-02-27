"""
    Helper callback functionality, not essential to research
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from torch import nn
from torch.optim import Optimizer


class FinetuneEncoderDecoder(BaseFinetuning):
    def __init__(
        self,
        unfreeze_encoder_at_epoch: int = 1,
        encoder_initial_lr_div: float = 10,
        train_batchnorm: bool = True,
    ):
        """
        Finetune a encoder model based on a learning rate.

        Args:

            unfreeze_encoder_at_epoch: Epoch at which the encoder will be unfreezed.

            encoder_initial_lr_div:
                Used to scale down the encoder learning rate compared to rest of model.

            train_batchnorm: Make Batch Normalization trainable at the beginning of train.
        """
        super().__init__()
        self.unfreeze_encoder_at_epoch = unfreeze_encoder_at_epoch
        self.encoder_initial_lr_div = encoder_initial_lr_div
        self.train_batchnorm = train_batchnorm

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Check if the LightningModule has the necessary attribute before the train starts

        Args:
            trainer: Lightning Trainer
            pl_module: Lightning Module used during train

        Raises:
            Exception: If LightningModule has no nn.Module `encoder` attribute.
        """
        if hasattr(pl_module, "encoder") and isinstance(pl_module.encoder, nn.Module):
            return
        raise Exception(
            "The LightningModule should have a nn.Module `encoder` attribute"
        )

    def freeze_before_training(self, pl_module: pl.LightningModule):
        """Freeze the encoder initially before the train starts.

        Args:
            pl_module: Lightning Module
        """
        self.freeze(pl_module.encoder, train_bn=self.train_batchnorm)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ):
        """Unfreezes the encoder at the specified epoch

        Args:
            pl_module: Lightning Module
            epoch: epoch number
            optimizer: optimizer used during training
            opt_idx: optimizer index
        """
        if epoch == self.unfreeze_encoder_at_epoch:
            self.unfreeze_and_add_param_group(
                pl_module.encoder,
                optimizer,
                initial_denom_lr=self.encoder_initial_lr_div,
                train_bn=not self.train_batchnorm,
            )
