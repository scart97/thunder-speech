# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torchaudio.datasets.utils import extract_archive

from thunder.quartznet.compatibility import (
    download_checkpoint,
    load_quartznet_weights,
    read_params_from_config,
)
from thunder.quartznet.model import Quartznet5, Quartznet_decoder
from thunder.quartznet.preprocess import FilterbankFeatures
from thunder.text_processing.transform import BatchTextTransformer
from thunder.text_processing.vocab import Vocab


class QuartznetModule(pl.LightningModule):
    def __init__(
        self,
        initial_vocab_tokens: List[str],
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        n_fft: int = 512,
        preemph: float = 0.97,
        nfilt: int = 64,
        dither: float = 1e-5,
        filters: List[int] = [256, 256, 512, 512, 512],
        kernel_sizes: List[int] = [33, 39, 51, 63, 75],
        repeat_blocks: int = 1,
        learning_rate: float = 3e-4,
        nemo_compat_vocab: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.features = FilterbankFeatures(
            sample_rate=sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=nfilt,
            dither=dither,
        )

        self.encoder = Quartznet5(nfilt, filters, kernel_sizes, repeat_blocks)

        vocab = Vocab(initial_vocab_tokens, nemo_compat=nemo_compat_vocab)
        self.decoder = self.build_decoder(1024, len(vocab))
        self.text_pipeline = BatchTextTransformer(vocab=vocab)
        self.loss_func = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)

    def build_decoder(self, decoder_input_channels: int, vocab_size: int):
        return Quartznet_decoder(
            num_classes=vocab_size, input_channels=decoder_input_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        encoded = self.encoder(features)
        return self.decoder(encoded)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> List[str]:
        pred = self(x)
        return self.text_pipeline.decode_prediction(pred)

    def process_texts(self, texts):
        return self.text_pipeline.encode(texts, device=self.device)

    def training_step(self, batch, batch_idx):
        audio, audio_lens, texts = batch
        y, y_lens = self.process_texts(texts)

        probs = self(audio)
        probs = probs.permute(2, 0, 1)  # NCT -> TNC
        logprobs = nn.LogSoftmax(2)(probs)
        audio_lens = (audio_lens * probs.shape[0]).long()
        loss = self.loss_func(logprobs, y, audio_lens, y_lens)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, audio_lens, texts = batch
        y, y_lens = self.process_texts(texts)

        probs = self(audio)
        probs = probs.permute(2, 0, 1)  # NCT -> TNC
        logprobs = nn.LogSoftmax(2)(probs)
        audio_lens = (audio_lens * probs.shape[0]).long()
        loss = self.loss_func(logprobs, y, audio_lens, y_lens)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

    @classmethod
    def load_from_nemo(cls, *, nemo_filepath: str = None, checkpoint_name: str = None):
        if checkpoint_name is not None:
            nemo_filepath = download_checkpoint(checkpoint_name)
        if nemo_filepath is None:
            raise ValueError("Either nemo_filepath or checkpoint_name must be passed")

        with TemporaryDirectory() as extract_path:
            extract_path = Path(extract_path)
            extract_archive(str(nemo_filepath), extract_path)
            config_path = extract_path / "model_config.yaml"
            encoder_params, initial_vocab, preprocess_params = read_params_from_config(
                config_path
            )
            module = cls(
                initial_vocab_tokens=initial_vocab,
                **encoder_params,
                **preprocess_params,
                nemo_compat_vocab=True,
            )
            weights_path = extract_path / "model_weights.ckpt"
            load_quartznet_weights(module.encoder, module.decoder, weights_path)
        # Here we set it in eval mode, so it correctly works during inference
        # Supposing that the majority of applications will be either load a checkpoint
        # and directly run inference, or fine-tuning. Either way this will prevent a silent
        # bug (case 1) or will be ignored (case 2).
        module.eval()
        return module
