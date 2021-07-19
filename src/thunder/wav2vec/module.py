# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = [
    "Wav2Vec2Module",
    "TextTransformConfig",
    "ModelConfig",
    "OptimizerConfig",
]

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

try:
    from transformers import Wav2Vec2Model
except ModuleNotFoundError as transformers_not_installed:
    raise ImportError(
        "To use wav2vec please install the transformers extension, by calling `pip install thunder-speech[transformers]`"
    ) from transformers_not_installed

from torchaudio.models.wav2vec2.utils.import_huggingface import _get_config, _get_model

from thunder.blocks import linear_decoder
from thunder.ctc_loss import calculate_ctc
from thunder.metrics import CER, WER
from thunder.text_processing.transform import BatchTextTransformer, TextTransformConfig
from thunder.wav2vec.transform import Wav2Vec2Preprocess


@dataclass
class ModelConfig:
    """Configuration to create the wav2vec 2.0 encoder.

    Attributes:
        model_name: Name of the original huggingface checkpoint to load from. defaults to 'facebook/wav2vec2-base'
        gradient_checkpointing: Use gradient checkpointing to save memory at the expense of slower backward pass. defaults to False.
        additional_kwargs: Any other option that can be passed to the original Wav2Vec2Model.from_pretrained. defaults to {}.
        decoder_dropout: Dropout before the final decoding layer. defaults to 0.1.
    """

    model_name: str = "facebook/wav2vec2-base"
    gradient_checkpointing: bool = False
    additional_kwargs: Dict[str, Any] = field(default_factory=lambda: copy({}))
    decoder_dropout: float = 0.1


@dataclass
class OptimizerConfig:
    """Configuration used by the optimizer

    Attributes:
        learning_rate: learning rate. defaults to 3e-4.
    """

    learning_rate: float = 3e-4


class Wav2Vec2Module(pl.LightningModule):
    def __init__(
        self,
        text_cfg: TextTransformConfig,
        encoder_cfg: ModelConfig = ModelConfig(),
        optim_cfg: OptimizerConfig = OptimizerConfig(),
    ):
        """Wav2Vec model for fine-tuning.

        Args:
            text_cfg: Configuration for the text processing pipeline
            encoder_cfg: Configuration for the wav2vec encoder
            optim_cfg: Configuration for the optimizer used during training
        """
        super().__init__()
        self.save_hyperparameters()
        self.audio_transform = Wav2Vec2Preprocess()

        self.encoder = Wav2Vec2Model.from_pretrained(
            encoder_cfg.model_name,
            gradient_checkpointing=encoder_cfg.gradient_checkpointing,
            **encoder_cfg.additional_kwargs,
        )
        self.encoder.feature_extractor._freeze_parameters()

        self.text_transform = BatchTextTransformer(text_cfg)
        self.decoder = linear_decoder(
            self.encoder.config.hidden_size,
            len(self.text_transform.vocab),
            encoder_cfg.decoder_dropout,
        )

        # Metrics
        self.val_cer = CER()
        self.val_wer = WER()
        # Example input is one second of fake audio
        self.example_input_array = torch.randn((10, 16000))

    def forward(self, audio: Tensor) -> Tensor:
        """Process the audio tensor to create the probabilities.

        Args:
            audio : Audio tensor of shape [batch_size, time]

        Returns:
            Tensor with the prediction probabilities.
        """
        features = self.audio_transform(audio)
        encoded_dict = self.encoder(features)
        return self.decoder(encoded_dict.last_hidden_state)

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
        )


class Wav2Vec2Scriptable(nn.Module):
    def __init__(self, module: Wav2Vec2Module, quantized: bool = False):
        """Wav2vec model ready to be jit scripted and used in inference.
        This class is necessary because the torchaudio imported model is not
        a drop in replacement of the transformers implementation, causing some
        trouble with the jit.

        Args:
            module : The trained Wav2Vec2 module that you want to export
            quantized : Controls if quantization will be applied to the model.
        """
        super().__init__()
        # Transforming model to torchaudio one
        encoder_config = _get_config(module.encoder.config)
        encoder_config["encoder_num_out"] = len(module.text_transform.vocab)
        imported = _get_model(**encoder_config)
        imported.feature_extractor.load_state_dict(
            module.encoder.feature_extractor.state_dict()
        )
        imported.encoder.feature_projection.load_state_dict(
            module.encoder.feature_projection.state_dict()
        )
        imported.encoder.transformer.load_state_dict(
            module.encoder.encoder.state_dict()
        )
        imported.encoder.readout.load_state_dict(module.decoder[1].state_dict())

        if quantized:
            imported.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
            imported = torch.quantization.quantize_dynamic(
                imported, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
            )

        self.model = imported

        self.audio_transform = module.audio_transform
        self.text_transform = module.text_transform

    def forward(self, audio: Tensor) -> Tensor:
        """Process the audio tensor to create the probabilities.

        Args:
            audio : Audio tensor of shape [batch_size, time]

        Returns:
            Tensor with the prediction probabilities.
        """
        features = self.audio_transform(audio)
        outputs = self.model(features)
        probs = outputs[0]
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
