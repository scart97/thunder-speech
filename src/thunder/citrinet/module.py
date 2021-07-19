"""Citrinet LightningModule that combines all of the individual parts
to enable easy finetuning and inference.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = [
    "CitrinetModule",
    "CitrinetCheckpoint",
    "TextTransformConfig",
    "EncoderConfig",
    "FilterbankConfig",
    "OptimizerConfig",
]

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchaudio.datasets.utils import extract_archive

from thunder.blocks import conv1d_decoder
from thunder.citrinet.blocks import Citrinet_encoder, EncoderConfig
from thunder.citrinet.compatibility import (
    CitrinetCheckpoint,
    fix_vocab,
    read_params_from_config_citrinet,
)
from thunder.ctc_loss import calculate_ctc
from thunder.metrics import CER, WER
from thunder.quartznet.compatibility import load_quartznet_weights
from thunder.quartznet.transform import FilterbankConfig, FilterbankFeatures
from thunder.text_processing.transform import BatchTextTransformer, TextTransformConfig
from thunder.utils import download_checkpoint


@dataclass
class OptimizerConfig:
    """Configuration used by the optimizer

    Attributes:
        learning_rate: learning rate. defaults to 3e-4.
        betas: beta1 and beta2 used by adam. defaults to (0.8, 0.5), similar to the novograd values on nemo.
    """

    learning_rate: float = 3e-4
    betas: Tuple[float] = (0.8, 0.5)


class CitrinetModule(pl.LightningModule):
    def __init__(
        self,
        text_cfg: TextTransformConfig,
        encoder_cfg: EncoderConfig,
        audio_cfg: FilterbankConfig = FilterbankConfig(),
        optim_cfg: OptimizerConfig = OptimizerConfig(),
    ):
        """Module containing both the basic citrinet model and helper functionality, such as
        feature creation and text processing.

        Args:
            text_cfg: Configuration for the text processing pipeline
            encoder_cfg: Configuration for the citrinet encoder
            audio_cfg: Configuration for the filterbank features applied to the input audio
            optim_cfg: Configuration for the optimizer used during training
        """
        super().__init__()
        self.save_hyperparameters()

        self.audio_transform = FilterbankFeatures(audio_cfg)

        self.encoder = Citrinet_encoder(encoder_cfg)

        self.text_transform = BatchTextTransformer(text_cfg)
        self.decoder = conv1d_decoder(640, num_classes=len(self.text_transform.vocab))

        # Metrics
        self.val_cer = CER()
        self.val_wer = WER()
        # Example input is one second of fake audio
        self.example_input_array = torch.randn((10, audio_cfg.sample_rate))

    def forward(self, x: Tensor) -> Tensor:
        """Process the audio tensor to create the predictions.

        Args:
            x : Audio tensor of shape [batch_size, time]

        Returns:
            Tuple with the predictions and output lengths. Notice that the ouput lengths are not normalized, they are a long tensor.
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
        decoded_targets = self.text_transform.decode_prediction(y)
        self.val_cer(decoded_preds, decoded_targets)
        self.val_wer(decoded_preds, decoded_targets)

        self.log("loss/val_loss", loss)
        self.log("metrics/cer", self.val_cer, on_epoch=True)
        self.log("metrics/wer", self.val_wer, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configuring optimizers. Check the original lightning docs for more info.

        Returns:
            Optimizer, and optionally the learning rate scheduler.
        """
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.optim_cfg.learning_rate,
            betas=self.hparams.optim_cfg.betas,
        )

    @classmethod
    def load_from_nemo(
        cls, checkpoint: Union[str, CitrinetCheckpoint], save_folder: str = None
    ) -> "CitrinetModule":
        """Load from the original nemo checkpoint.

        Args:
            checkpoint : Path to local .nemo file or checkpoint to be downloaded locally and lodaded.
            save_folder : Path to save the checkpoint when downloading it. Ignored if you pass a .nemo file as the first argument.

        Returns:
            The model loaded from the checkpoint
        """
        if isinstance(checkpoint, CitrinetCheckpoint):
            nemo_filepath = download_checkpoint(checkpoint, save_folder)
        else:
            nemo_filepath = checkpoint

        with TemporaryDirectory() as extract_path:
            extract_archive(str(nemo_filepath), extract_path)
            extract_path = Path(extract_path)
            config_path = extract_path / "model_config.yaml"
            (
                encoder_params,
                initial_vocab,
                preprocess_params,
            ) = read_params_from_config_citrinet(config_path)

            sentencepiece_path = str(extract_path / "tokenizer.model")

            module = cls(
                TextTransformConfig(
                    initial_vocab_tokens=fix_vocab(initial_vocab),
                    sentencepiece_model=sentencepiece_path,
                    simple_vocab=True,
                ),
                EncoderConfig(**encoder_params),
                FilterbankConfig(**preprocess_params),
            )
            weights_path = extract_path / "model_weights.ckpt"
            load_quartznet_weights(module.encoder, module.decoder, str(weights_path))
        # Here we set it in eval mode, so it correctly works during inference
        # Supposing that the majority of applications will be either (1) load a checkpoint
        # and directly run inference, or (2) fine-tuning. Either way this will prevent a silent
        # bug (case 1) or will be ignored (case 2).
        module.eval()
        return module

    def change_vocab(self, text_cfg: TextTransformConfig):
        """Changes the vocabulary of the model. useful when finetuning to another language.

        Args:
            text_cfg: Configuration for the text processing pipeline
        """
        # Updating hparams so that the saved model can be correctly loaded
        self.hparams.text_cfg = text_cfg
        self.text_transform = BatchTextTransformer(text_cfg)
        self.decoder = conv1d_decoder(640, num_classes=len(self.text_transform.vocab))
