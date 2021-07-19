# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This test is modified from the pytorch lightning test
# source: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/boring_model.py

import pytest

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from thunder.callbacks import FinetuneEncoderDecoder


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(pl.LightningModule):
    def __init__(self):
        """
        Testing PL Module
        Use as follows:
        - subclass
        - modify the behavior for what you want
        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing
        or:
        model = BaseTestModel()
        model.training_epoch_end = None
        This is the same used by pytorch lightning to test
        """
        super().__init__()
        self.encoder = torch.nn.Linear(32, 5)
        self.decoder = torch.nn.Linear(5, 2)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()), lr=0.1
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


def test_finetune_callback():
    module = BoringModel()
    original_weight = module.encoder.weight.detach().clone()
    # Don't unfreeze before the time
    trainer = pl.Trainer(
        callbacks=[FinetuneEncoderDecoder(unfreeze_encoder_at_epoch=2)], max_epochs=1
    )
    trainer.fit(module)
    assert torch.allclose(module.encoder.weight, original_weight)
    # Assert that it changes after the freeze
    trainer2 = pl.Trainer(
        callbacks=[FinetuneEncoderDecoder(unfreeze_encoder_at_epoch=1)], max_epochs=3
    )
    trainer2.fit(module)
    assert not torch.allclose(module.encoder.weight, original_weight)


def test_callback_raises():
    module = BoringModel()
    del module.encoder
    with pytest.raises(Exception, match="encoder"):
        trainer = pl.Trainer(
            callbacks=[FinetuneEncoderDecoder(unfreeze_encoder_at_epoch=2)],
            max_epochs=1,
        )
        trainer.fit(module)
