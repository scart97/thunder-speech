"""Helper functions to load huggingface speech recognition models.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from transformers import AutoModelForCTC, Wav2Vec2Processor

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
        return (
            out.last_hidden_state,
            self.original_encoder._get_feat_extract_output_lengths(audio_lengths),
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
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    vocab = list(processor.tokenizer.get_vocab().keys())
    text_transform = BatchTextTransformer(
        tokens=vocab,
        blank_token=processor.tokenizer.pad_token,
        pad_token=processor.tokenizer.pad_token,
        unknown_token=processor.tokenizer.unk_token,
        start_token=processor.tokenizer.bos_token,
        end_token=processor.tokenizer.eos_token,
    )
    decoder = linear_decoder(
        model.base_model.config.hidden_size, len(vocab), decoder_dropout=0.0
    )
    if hasattr(model, "lm_head"):
        decoder[1].load_state_dict(model.lm_head.state_dict())

    module = BaseCTCModule(
        encoder=_HuggingFaceEncoderAdapt(
            model.base_model,
            mask_input=processor.feature_extractor.return_attention_mask,
        ),
        decoder=decoder,
        text_transform=text_transform,
        audio_transform=Wav2Vec2Preprocess(
            mask_input=processor.feature_extractor.return_attention_mask,
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
        Modified module ready to call torch.jit.script(module) or module.to_torchscript()
    """
    imported = import_huggingface_model(module.encoder.original_encoder)
    if quantized:
        imported.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        imported = torch.quantization.quantize_dynamic(
            imported, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
    module.encoder = imported
    return module
