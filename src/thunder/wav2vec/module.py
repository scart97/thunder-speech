# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["Wav2Vec2Module"]

from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn.functional import ctc_loss, log_softmax
from transformers import Wav2Vec2Model

from thunder.metrics import CER, WER
from thunder.text_processing.transform import BatchTextTransformer, Vocab
from thunder.wav2vec.transform import Wav2Vec2Preprocess


class Wav2Vec2Module(pl.LightningModule):
    def __init__(
        self,
        initial_vocab_tokens: List[str],
        model_name: str = "facebook/wav2vec2-base",
        gradient_checkpointing: bool = False,
        decoder_dropout: float = 0.1,
        learning_rate: float = 3e-4,
        **kwargs: Dict[str, Any],
    ):
        """Wav2Vec model for fine-tuning.

        Args:
            initial_vocab_tokens : List of tokens to be used in the vocab, special tokens should not be included here. Check [`docs`](https://scart97.github.io/thunder-speech/quick%20reference%20guide/#how-to-get-the-initial_vocab_tokens-from-my-dataset)
            model_name : Name of the original huggingface checkpoint to load from.
            gradient_checkpointing : Use gradient checkpointing to save memory at the expense of slower backward pass.
            decoder_dropout : Dropout before the final decoding layer
            learning_rate : Learning rate used on the optimizer.
            kwargs: Any other option that can be passed to the original Wav2Vec2Model.from_pretrained
        """
        super().__init__()
        self.save_hyperparameters()
        self.audio_transform = Wav2Vec2Preprocess()

        self.encoder = Wav2Vec2Model.from_pretrained(
            model_name,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.encoder.feature_extractor._freeze_parameters()

        self.text_transform = self.build_text_transform(initial_vocab_tokens)
        self.decoder = self.build_decoder(
            decoder_dropout,
            self.encoder.config.hidden_size,
            len(self.text_transform.vocab),
        )

        # Metrics
        self.val_cer = CER()
        self.val_wer = WER()
        # Example input is one second of fake audio
        self.example_input_array = torch.randn((10, 16000))

    def build_text_transform(
        self, initial_vocab_tokens: List[str]
    ) -> BatchTextTransformer:
        """Overwrite this function if you want to change how the text processing happens inside the model.

        Args:
            initial_vocab_tokens : List of tokens to create the vocabulary, special tokens should not be included here.

        Returns:
            The transform that will both `encode` the text and `decode_prediction`.
        """
        vocab = Vocab(initial_vocab_tokens)
        return BatchTextTransformer(vocab=vocab)

    def build_decoder(
        self, decoder_dropout: float, decoder_input_channels: int, vocab_size: int
    ) -> nn.Module:
        """Overwrite this function if you want to change the model decoder.

        Args:
            decoder_dropout: Amount of dropout to be used in the decoder
            decoder_input_channels : Number of input channels of the decoder. That is the number of channels of the features created by the encoder.
            vocab_size : Number of output classes

        Returns:
            Module that represents the decoder.
        """
        return nn.Sequential(
            nn.Dropout(decoder_dropout),
            nn.Linear(decoder_input_channels, vocab_size),
        )

    def forward(self, audio: Tensor) -> Tensor:
        """Process the audio tensor to create the probabilities.

        Args:
            audio : Audio tensor of shape [batch_size, time]

        Returns:
            Tensor with the prediction probabilities.
        """
        features = self.audio_transform(audio)
        encoded_dict = self.encoder(features)
        probs = self.decoder(encoded_dict.last_hidden_state)
        # Change from (batch, time, #vocab) to (batch, #vocab, time)
        # that is expected by the rest of the library
        return probs.permute(0, 2, 1)

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

    def calculate_loss(self, probabilities, y, prob_lens, y_lens):
        # Change from (batch, #vocab, time) to (time, batch, #vocab)
        probabilities = probabilities.permute(2, 0, 1)
        logprobs = log_softmax(probabilities, dim=2)
        # Calculate the logprobs correct length based on the
        # normalized original lengths
        prob_lens = (prob_lens * logprobs.shape[0]).long()
        blank = self.text_transform.vocab.blank_idx

        return ctc_loss(
            logprobs,
            y,
            prob_lens,
            y_lens,
            blank=blank,
            reduction="mean",
            zero_infinity=True,
        )

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
        loss = self.calculate_loss(probabilities, y, audio_lens, y_lens)

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
        loss = self.calculate_loss(probabilities, y, audio_lens, y_lens)

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
        )
