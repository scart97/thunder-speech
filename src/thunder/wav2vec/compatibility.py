try:
    from transformers import AutoModelForCTC, Wav2Vec2Processor
except ModuleNotFoundError as transformers_not_installed:
    raise ImportError(
        "To use any huggingface model please install the transformers extension, by calling `pip install thunder-speech[transformers]`"
    ) from transformers_not_installed

from typing import List

import torch
from torch import Tensor, nn
from torchaudio.models.wav2vec2.utils.import_huggingface import _get_config, _get_model

from thunder.blocks import linear_decoder
from thunder.text_processing.transform import BatchTextTransformer
from thunder.utils import CheckpointResult
from thunder.wav2vec.transform import Wav2Vec2Preprocess


class HuggingFaceEncoderAdapt(nn.Module):
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
        encoder=HuggingFaceEncoderAdapt(model.base_model),
        decoder=decoder,
        text_transform=text_transform,
        audio_transform=Wav2Vec2Preprocess(),
        encoder_final_dimension=model.base_model.config.hidden_size,
    )


class Wav2Vec2Scriptable(nn.Module):
    def __init__(self, module, quantized: bool = False):
        """Wav2vec model ready to be jit scripted and used in inference.
        This class is necessary because the torchaudio imported model is not
        a drop in replacement of the transformers implementation, causing some
        trouble with the jit.

        Args:
            module : The trained Wav2Vec2 LightningModule that you want to export
            quantized : Controls if quantization will be applied to the model.
        """
        super().__init__()
        # Transforming model to torchaudio one
        encoder = module.encoder.original_encoder
        encoder_config = _get_config(encoder.config)
        encoder_config["encoder_num_out"] = module.text_transform.num_tokens
        imported = _get_model(**encoder_config)
        imported.feature_extractor.load_state_dict(
            encoder.feature_extractor.state_dict()
        )
        imported.encoder.feature_projection.load_state_dict(
            encoder.feature_projection.state_dict()
        )
        imported.encoder.transformer.load_state_dict(encoder.encoder.state_dict())
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
