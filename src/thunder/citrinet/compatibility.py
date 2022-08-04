"""Helper functions to load the Citrinet model from original Nemo released checkpoint files.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["CitrinetCheckpoint", "load_components_from_citrinet_config", "fix_vocab"]


from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, TypedDict, Union

from omegaconf import OmegaConf
from torch import nn
from torchaudio.datasets.utils import extract_archive

from thunder.blocks import conv1d_decoder
from thunder.citrinet.blocks import CitrinetEncoder
from thunder.module import BaseCTCModule
from thunder.quartznet.compatibility import load_quartznet_weights
from thunder.quartznet.transform import FilterbankFeatures
from thunder.text_processing.transform import BatchTextTransformer
from thunder.utils import BaseCheckpoint, download_checkpoint


# fmt:off
class CitrinetCheckpoint(BaseCheckpoint):
    """Trained model weight checkpoints.
    Used by [`download_checkpoint`][thunder.utils.download_checkpoint] and
    [`load_citrinet_checkpoint`][thunder.citrinet.compatibility.load_citrinet_checkpoint].

    Note:
        Possible values are `stt_en_citrinet_256`,`stt_en_citrinet_512`,`stt_en_citrinet_1024`, `stt_es_citrinet_512`
    """
    stt_en_citrinet_256 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256/versions/1.0.0rc1/files/stt_en_citrinet_256.nemo"
    stt_en_citrinet_512 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512/versions/1.0.0rc1/files/stt_en_citrinet_512.nemo"
    stt_en_citrinet_1024 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024/versions/1.0.0rc1/files/stt_en_citrinet_1024.nemo"
    stt_es_citrinet_512 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_citrinet_512/versions/1.0.0/files/stt_es_citrinet_512.nemo"
# fmt:on


class AugmentParams(TypedDict, total=False):
    num_cutout_masks: int
    num_time_masks: int
    num_freq_masks: int
    mask_time_width: int
    mask_freq_width: int
    dropout: float


def load_components_from_citrinet_config(
    config_path: Union[str, Path],
    sentencepiece_path: Union[str, Path],
    augment_params: AugmentParams = None,
) -> Tuple[nn.Module, nn.Module, BatchTextTransformer]:
    """Read the important parameters from the config stored inside the .nemo
    checkpoint.

    Args:
        config_path: Path to the .yaml file, usually called model_config.yaml
        sentencepiece_path: Path to the sentencepiece model used to tokenize, usually called tokenizer.model

    Returns:
        A tuple containing, in this order, the encoder, the audio transform and the text transform
    """
    augment_params = augment_params or {}

    conf = OmegaConf.load(config_path)
    encoder_params = conf["encoder"]
    quartznet_conf = OmegaConf.to_container(encoder_params["jasper"])

    body_config = quartznet_conf[1:-1]

    filters = [cfg["filters"] for cfg in body_config]
    kernel_sizes = [cfg["kernel"][0] for cfg in body_config]
    strides = [cfg["stride"][0] for cfg in body_config]
    encoder_cfg = {
        "filters": filters,
        "kernel_sizes": kernel_sizes,
        "strides": strides,
        "dropout": augment_params.pop("dropout", 0.0),
    }
    preprocess = conf["preprocessor"]

    preprocess_cfg = {
        "sample_rate": preprocess["sample_rate"],
        "n_window_size": int(preprocess["window_size"] * preprocess["sample_rate"]),
        "n_window_stride": int(preprocess["window_stride"] * preprocess["sample_rate"]),
        "n_fft": preprocess["n_fft"],
        "nfilt": preprocess["features"],
        "dither": preprocess["dither"],
        **augment_params,
    }

    labels = conf["labels"] if "labels" in conf else conf["decoder"]["vocabulary"]

    encoder = CitrinetEncoder(**encoder_cfg)
    text_transform = BatchTextTransformer(
        tokens=fix_vocab(labels),
        sentencepiece_model=sentencepiece_path,
    )
    audio_transform = FilterbankFeatures(**preprocess_cfg)

    return (
        encoder,
        audio_transform,
        text_transform,
    )


def fix_vocab(vocab_tokens: List[str]) -> List[str]:
    """Transform the nemo vocab tokens back to the sentencepiece sytle
    with the _ prefix

    Args:
        vocab_tokens: List of tokens in the vocabulary

    Returns:
        New list of tokens with the new prefix
    """
    out_tokens = []
    for token in vocab_tokens:
        if token.startswith("##"):
            out_tokens.append(token[2:])
        else:
            out_tokens.append("▁" + token)
    return out_tokens


def load_citrinet_checkpoint(
    checkpoint: Union[str, CitrinetCheckpoint],
    save_folder: str = None,
    augment_params: AugmentParams = None,
) -> BaseCTCModule:
    """Load from the original nemo checkpoint.

    Args:
        checkpoint: Path to local .nemo file or checkpoint to be downloaded locally and lodaded.
        save_folder: Path to save the checkpoint when downloading it. Ignored if you pass a .nemo file as the first argument.

    Returns:
        The model loaded from the checkpoint
    """
    if isinstance(checkpoint, CitrinetCheckpoint):
        nemo_filepath = download_checkpoint(checkpoint, save_folder)
    else:
        nemo_filepath = checkpoint

    with TemporaryDirectory() as extract_folder:
        extract_archive(str(nemo_filepath), extract_folder)
        extract_path = Path(extract_folder)
        config_path = extract_path / "model_config.yaml"
        sentencepiece_path = str(extract_path / "tokenizer.model")
        (
            encoder,
            audio_transform,
            text_transform,
        ) = load_components_from_citrinet_config(
            config_path, sentencepiece_path, augment_params
        )

        decoder = conv1d_decoder(640, num_classes=text_transform.num_tokens)

        weights_path = extract_path / "model_weights.ckpt"
        load_quartznet_weights(encoder, decoder, str(weights_path))
        module = BaseCTCModule(
            encoder,
            decoder,
            audio_transform,
            text_transform,
            encoder_final_dimension=640,
        )
        return module.eval()
