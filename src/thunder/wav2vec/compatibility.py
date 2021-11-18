try:
    from transformers import AutoModelForCTC, Wav2Vec2Processor
except ModuleNotFoundError as transformers_not_installed:
    raise ImportError(
        "To use any huggingface model please install the transformers extension, by calling `pip install thunder-speech[transformers]`"
    ) from transformers_not_installed

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torchaudio.models.wav2vec2.utils import import_huggingface_model

from thunder.blocks import linear_decoder
from thunder.text_processing.transform import BatchTextTransformer
from thunder.utils import CheckpointResult
from thunder.wav2vec.transform import Wav2Vec2Preprocess


class _HuggingFaceEncoderAdapt(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.original_encoder = encoder
        if hasattr(self.original_encoder, "freeze_feature_extractor"):
            self.original_encoder.freeze_feature_extractor()

    def forward(self, x):
        out = self.original_encoder(x)
        return out.last_hidden_state


def load_huggingface_checkpoint(model_name: str, **model_kwargs) -> CheckpointResult:
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

    return CheckpointResult(
        encoder=_HuggingFaceEncoderAdapt(model.base_model),
        decoder=decoder,
        text_transform=text_transform,
        audio_transform=Wav2Vec2Preprocess(),
        encoder_final_dimension=model.base_model.config.hidden_size,
    )


class _GetFirstElement(nn.Module):
    def forward(self, x: Tuple[Tensor, Optional[Tensor]]) -> Tensor:
        return x[0]


def prepare_scriptable_wav2vec(module, quantized: bool = False):
    imported = import_huggingface_model(module.encoder.original_encoder)
    if quantized:
        imported.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        imported = torch.quantization.quantize_dynamic(
            imported, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
    module.encoder = nn.Sequential(imported, _GetFirstElement())
    return module
