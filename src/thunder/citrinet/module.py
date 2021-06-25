"""Citrinet LightningModule that combines all of the individual parts
to enable easy finetuning and inference.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["CitrinetModule", "CitrinetCheckpoint"]

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchaudio.datasets.utils import extract_archive

from thunder.blocks import conv1d_decoder
from thunder.citrinet.blocks import Citrinet_encoder
from thunder.citrinet.compatibility import (
    CitrinetCheckpoint,
    fix_vocab,
    read_params_from_config_citrinet,
)
from thunder.ctc_loss import calculate_ctc
from thunder.metrics import CER, WER
from thunder.quartznet.compatibility import load_quartznet_weights
from thunder.quartznet.transform import FilterbankFeatures
from thunder.text_processing.transform import BatchTextTransformer
from thunder.utils import download_checkpoint


class CitrinetModule(pl.LightningModule):
    def __init__(
        self,
        initial_vocab_tokens: List[str],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        sentencepiece_model: Optional[str] = None,
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        n_fft: int = 512,
        preemph: float = 0.97,
        nfilt: int = 64,
        dither: float = 1e-5,
        learning_rate: float = 3e-4,
        nemo_compat_vocab: bool = False,
    ):
        """Module containing both the basic quartznet model and helper functionality, such as
        feature creation and text processing.

        Args:
            initial_vocab_tokens : List of tokens to be used in the vocab, special tokens should not be included here. Check [`docs`](https://scart97.github.io/thunder-speech/quick%20reference%20guide/#how-to-get-the-initial_vocab_tokens-from-my-dataset)
            filters : Check [`Citrinet_encoder`][thunder.citrinet.blocks.Citrinet_encoder]
            kernel_sizes : Check [`Citrinet_encoder`][thunder.citrinet.blocks.Citrinet_encoder]
            strides : Check [`Citrinet_encoder`][thunder.citrinet.blocks.Citrinet_encoder]
            sample_rate : Check [`FilterbankFeatures`][thunder.quartznet.transform.FilterbankFeatures]
            n_window_size : Check [`FilterbankFeatures`][thunder.quartznet.transform.FilterbankFeatures]
            n_window_stride : Check [`FilterbankFeatures`][thunder.quartznet.transform.FilterbankFeatures]
            n_fft : Check [`FilterbankFeatures`][thunder.quartznet.transform.FilterbankFeatures]
            preemph : Check [`FilterbankFeatures`][thunder.quartznet.transform.FilterbankFeatures]
            nfilt : Check [`FilterbankFeatures`][thunder.quartznet.transform.FilterbankFeatures]
            dither : Check [`FilterbankFeatures`][thunder.quartznet.transform.FilterbankFeatures]
            learning_rate : Learning rate used by the optimizer
            nemo_compat_vocab : Controls if the used vocabulary will be compatible with the original nemo implementation.
        """
        super().__init__()
        self.save_hyperparameters()

        self.audio_transform = FilterbankFeatures(
            sample_rate=sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=nfilt,
            dither=dither,
        )

        self.encoder = Citrinet_encoder(nfilt, filters, kernel_sizes, strides)

        self.text_transform = BatchTextTransformer(
            initial_vocab_tokens, nemo_compat_vocab, sentencepiece_model
        )
        self.decoder = conv1d_decoder(640, num_classes=len(self.text_transform.vocab))

        # Metrics
        self.val_cer = CER()
        self.val_wer = WER()
        # Example input is one second of fake audio
        self.example_input_array = torch.randn((10, sample_rate))

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
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.5),
        )

    @classmethod
    def load_from_nemo(
        cls, *, nemo_filepath: str = None, checkpoint_name: CitrinetCheckpoint = None
    ) -> "CitrinetModule":
        """Load from the original nemo checkpoint.

        Args:
            nemo_filepath : Path to local .nemo file.
            checkpoint_name : Name of checkpoint to be downloaded locally and lodaded.

        Raises:
            ValueError: You need to pass only one of the two parameters.

        Returns:
            The model loaded from the checkpoint
        """
        if checkpoint_name is not None:
            nemo_filepath = download_checkpoint(checkpoint_name)
        if nemo_filepath is None and checkpoint_name is None:
            raise ValueError("Either nemo_filepath or checkpoint_name must be passed")

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
                initial_vocab_tokens=fix_vocab(initial_vocab),
                sentencepiece_model=sentencepiece_path,
                **encoder_params,
                **preprocess_params,
                nemo_compat_vocab=True,
            )
            weights_path = extract_path / "model_weights.ckpt"
            load_quartznet_weights(module.encoder, module.decoder, str(weights_path))
        # Here we set it in eval mode, so it correctly works during inference
        # Supposing that the majority of applications will be either (1) load a checkpoint
        # and directly run inference, or (2) fine-tuning. Either way this will prevent a silent
        # bug (case 1) or will be ignored (case 2).
        module.eval()
        return module

    def change_vocab(
        self,
        new_vocab_tokens: List[str],
        sentencepiece_model: Optional[str] = None,
        nemo_compat: bool = False,
    ):
        """Changes the vocabulary of the model. useful when finetuning to another language.

        Args:
            new_vocab_tokens : List of tokens to be used in the vocabulary, special tokens should not be included here.
            sentencepiece_model: path to the sentencepiece `tokenizer.model` file
            nemo_compat : Controls if the used vocabulary will be compatible with the original nemo implementation.
        """
        # Updating hparams so that the saved model can be correctly loaded
        self.hparams.initial_vocab_tokens = new_vocab_tokens
        self.hparams.nemo_compat_vocab = nemo_compat
        self.hparams.sentencepiece_model = sentencepiece_model

        self.text_transform = BatchTextTransformer(
            new_vocab_tokens, nemo_compat, sentencepiece_model
        )
        self.decoder = conv1d_decoder(640, num_classes=len(self.text_transform.vocab))
