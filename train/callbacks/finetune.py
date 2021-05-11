import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

from torch.optim import Optimizer
from torch import nn


class FinetuneEncoderDecoder(BaseFinetuning):
    # First epoch is 0, the default unfreezes at the second one
    def __init__(
        self,
        unfreeze_encoder_at_epoch: int = 1,
        encoder_initial_lr_div: float = 10,
        train_bn: bool = True,
    ):
        """
        Finetune a encoder model based on a learning rate.

        Args:

            unfreeze_encoder_at_epoch: Epoch at which the encoder will be unfreezed.

            encoder_initial_lr_div:
                Used to scale down the encoder learning rate compared to rest of model.

            train_bn: Wheter to make Batch Normalization trainable.
        """
        super().__init__()
        self.unfreeze_encoder_at_epoch = unfreeze_encoder_at_epoch
        self.encoder_initial_lr_div = encoder_initial_lr_div
        self.train_bn = train_bn

    def on_fit_start(self, trainer, pl_module):
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `encoder` attribute.
        """
        if hasattr(pl_module, "encoder") and isinstance(pl_module.encoder, nn.Module):
            return
        raise Exception("The LightningModule should have a nn.Module `encoder` attribute")

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(pl_module.encoder, train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ):
        if epoch == self.unfreeze_encoder_at_epoch:
            print("num param groups:", len(optimizer.param_groups))
            optimizer.param_groups[-1]["lr"] = 1e-5
            print("Unfreezing and setting lr to...")
            self.unfreeze_and_add_param_group(
                pl_module.encoder,
                optimizer,
                initial_denom_lr=self.encoder_initial_lr_div,
                train_bn=not self.train_bn,
            )
