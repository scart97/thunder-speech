"""Helper functions to load huggingface speech recognition models.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from torch import Tensor, nn
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

from thunder.blocks import lengths_to_mask, linear_decoder
from thunder.huggingface.transform import Wav2Vec2Preprocess
from thunder.module import BaseCTCModule
from thunder.text_processing.transform import BatchTextTransformer


class _HuggingFaceEncoderAdapt(nn.Module):
    def __init__(self, encoder, mask_input: bool = False):
        super().__init__()
        self.original_encoder = encoder
        if hasattr(self.original_encoder, "freeze_feature_encoder"):
            self.original_encoder.freeze_feature_encoder()
        self.mask_input = mask_input

    def forward(self, audio: Tensor, audio_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        attention_mask: Optional[Tensor] = None
        if self.mask_input:
            attention_mask = lengths_to_mask(
                audio_lengths, max_length=audio.size(-1)
            ).int()
        out = self.original_encoder(audio, attention_mask=attention_mask)
        out_correct_shape = out.last_hidden_state.transpose(-1, -2)
        return (
            out_correct_shape,
            self.original_encoder._get_feat_extract_output_lengths(audio_lengths),
        )


def _get_special_token(tokenizer: AutoTokenizer, token_name: str):
    token = getattr(tokenizer, token_name)
    if token in tokenizer.additional_special_tokens:
        return None
    return token


def _tok_to_transform(tokenizer: AutoTokenizer) -> BatchTextTransformer:
    vocab = [v if v != "|" else " " for v in tokenizer.get_vocab().keys()]
    # Remove tokens that were added after the model was trained
    for t in tokenizer.additional_special_tokens:
        vocab.remove(t)
    return BatchTextTransformer(
        tokens=vocab,
        blank_token=_get_special_token(tokenizer, "pad_token"),
        pad_token=_get_special_token(tokenizer, "pad_token"),
        unknown_token=_get_special_token(tokenizer, "unk_token"),
    )


def load_huggingface_checkpoint(
    model_name: str, **model_kwargs: Dict[str, Any]
) -> BaseCTCModule:
    """Load huggingface model and convert to thunder [`BaseCTCModule`][thunder.module.BaseCTCModule]

    Args:
        model_name: huggingface identifier of the model, like "facebook/wav2vec2-large-960h"
        model_kwargs: extra keyword arguments to be passed to `AutoModelForCTC.from_pretrained`

    Returns:
        Thunder module containing the huggingface model.
    """
    model = AutoModelForCTC.from_pretrained(model_name, **model_kwargs)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # Some models only contain the encoder, and no tokenizer
    # In that case we need to warn the user to fix it before training
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_transform = _tok_to_transform(tokenizer)
        decoder = linear_decoder(
            model.base_model.config.hidden_size,
            text_transform.num_tokens,
            decoder_dropout=0.0,
        )
        if hasattr(model, "lm_head"):
            decoder[2].load_state_dict(model.lm_head.state_dict())
    except (OSError, KeyError):
        warn(
            UserWarning(
                "Huggingface model is missing the tokenizer! decoder and text_transform were not initialized"
            )
        )
        text_transform = None
        decoder = None

    module = BaseCTCModule(
        encoder=_HuggingFaceEncoderAdapt(
            model.base_model,
            mask_input=feature_extractor.return_attention_mask,
        ),
        decoder=decoder,
        text_transform=text_transform,
        audio_transform=Wav2Vec2Preprocess(
            mask_input=feature_extractor.return_attention_mask,
        ),
        encoder_final_dimension=model.base_model.config.hidden_size,
    )
    return module.eval()


def prepare_scriptable_wav2vec(
    module: BaseCTCModule, quantized: bool = False
) -> BaseCTCModule:
    """Converts thunder module containing a wav2vec2 model to be scriptable.

    Args:
        module: Module containing wav2vec2
        quantized: If true, also performs quantization of the model

    Returns:
        Modified module ready to call module.to_torchscript()
    """
    imported = import_huggingface_model(module.encoder.original_encoder)
    if quantized:
        imported.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        imported = torch.quantization.quantize_dynamic(
            imported, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
    module.encoder = imported
    module.decoder = nn.Sequential(*module.decoder[1:])
    return module
