# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytorch_lightning as pl
import torch
from torch.nn.functional import log_softmax, ctc_loss
from torch.nn import CTCLoss
from torchaudio.datasets.utils import extract_archive
from thunder.metrics import CER, WER
from thunder.quartznet.compatibility import (
    download_checkpoint,
    load_quartznet_weights,
    read_params_from_config,
)
from thunder.quartznet.model import Quartznet5, Quartznet_decoder
from thunder.quartznet.preprocess import FilterbankFeatures
from thunder.text_processing.transform import BatchTextTransformer
from thunder.text_processing.vocab import Vocab
import wandb


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
        self.n_window_stride = n_window_stride
        self.encoder = Quartznet5(nfilt, filters, kernel_sizes, repeat_blocks)
        self.converged = False
        self.text_pipeline = self.build_text_pipeline(initial_vocab_tokens, nemo_compat_vocab)
        self.decoder = self.build_decoder(1024, len(self.text_pipeline.vocab))
        # Metrics
        self.val_cer = CER()
        self.val_wer = WER()
        # self.example_input_array = torch.randn((16, sample_rate))
        self.example_input_array = None

    def build_text_pipeline(
        self, initial_vocab_tokens: List[str], nemo_compat_vocab: bool
    ) -> BatchTextTransformer:
        vocab = Vocab(initial_vocab_tokens, nemo_compat=nemo_compat_vocab)
        return BatchTextTransformer(vocab=vocab)

    def build_decoder(self, decoder_input_channels: int, vocab_size: int):
        return Quartznet_decoder(num_classes=vocab_size, input_channels=decoder_input_channels)

    def forward(self, x: torch.Tensor, audio_lens: torch.Tensor = None) -> torch.Tensor:
        max_samples = x.shape[-1]
        seq_lens = torch.floor((audio_lens * max_samples) / self.n_window_stride) + 1
        features = self.features(x, seq_lens)

        torch.set_printoptions(threshold=10_000)

        encoded, encoded_lens = self.encoder(features, seq_lens)
        encoded_lens = encoded_lens.long()
        decoded = self.decoder(encoded)

        return decoded, encoded_lens

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> List[str]:
        pred = self(x)
        return self.text_pipeline.decode_prediction(pred.argmax(1))

    def calculate_loss(self, probabilities, y, encoded_lens, y_lens):
        # Change from (batch, #vocab, time) to (time, batch, #vocab)
        probabilities = probabilities.permute(2, 0, 1)
        logprobs = log_softmax(probabilities, dim=2)
        torch.set_printoptions(threshold=10_000)
        blank = self.text_pipeline.vocab.blank_idx
        loss = torch.mean(
            ctc_loss(
                logprobs,
                y,
                encoded_lens,
                y_lens,
                blank=blank,
                reduction="none",
                zero_infinity=False,
            )
        )
        return loss

    def training_step(self, batch, batch_idx):
        audio, audio_lens, texts = batch

        y, y_lens = self.text_pipeline.encode(texts, device=self.device)
        probabilities, encoded_lens = self(audio, audio_lens)
        loss = self.calculate_loss(probabilities, y, encoded_lens, y_lens)

        self.log("loss/train_loss", loss)
        metrics = {"loss": loss}
        return loss

    def validation_step(self, batch, batch_idx):
        audio, audio_lens, texts = batch
        y, y_lens = self.text_pipeline.encode(texts, device=self.device)

        probabilities, encoded_lens = self(audio, audio_lens)
        loss = self.calculate_loss(probabilities, y, encoded_lens, y_lens)

        decoded_preds = self.text_pipeline.decode_prediction(probabilities.argmax(1))
        decoded_targets = self.text_pipeline.decode_prediction(y)
        self.val_cer(decoded_preds, decoded_targets)
        self.val_wer(decoded_preds, decoded_targets)

        self.log("loss/val_loss", loss)
        self.log("metrics/cer", self.val_cer, on_epoch=True)
        self.log("metrics/wer", self.val_wer, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
        )
        return opt

    @classmethod
    def load_from_nemo(
        cls,
        *,
        nemo_filepath: str = None,
        checkpoint_name: str = None,
    ):
        if checkpoint_name is not None:
            nemo_filepath = download_checkpoint(checkpoint_name)
        if nemo_filepath is None:
            raise ValueError("Either nemo_filepath or checkpoint_name must be passed")

        with TemporaryDirectory() as extract_path:
            extract_path = Path(extract_path)
            extract_archive(str(nemo_filepath), extract_path)
            config_path = extract_path / "model_config.yaml"
            encoder_params, initial_vocab, preprocess_params = read_params_from_config(config_path)
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

    def change_vocab(self, new_vocab_tokens: List[str]):
        # Updating hparams so that the saved model can be correctly loaded
        self.hparams.initial_vocab_tokens = new_vocab_tokens
        self.hparams.nemo_compat_vocab = True
        # Changed the last arg to True, it is "nemo_compat_vocab"
        self.text_pipeline = self.build_text_pipeline(new_vocab_tokens, True)
        self.decoder = self.build_decoder(1024, len(self.text_pipeline.vocab))
