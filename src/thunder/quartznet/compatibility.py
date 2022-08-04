"""Helper functions to load the Quartznet model from original Nemo released checkpoint files.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = [
    "QuartznetCheckpoint",
    "load_components_from_quartznet_config",
    "load_quartznet_weights",
    "load_quartznet_checkpoint",
]

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, TypedDict, Union

import torch
from omegaconf import OmegaConf
from torch import nn
from torchaudio.datasets.utils import extract_archive

from thunder.blocks import conv1d_decoder
from thunder.module import BaseCTCModule
from thunder.quartznet.blocks import QuartznetEncoder
from thunder.quartznet.transform import FilterbankFeatures
from thunder.text_processing.transform import BatchTextTransformer
from thunder.utils import BaseCheckpoint, download_checkpoint


# fmt:off
class QuartznetCheckpoint(BaseCheckpoint):
    """Trained model weight checkpoints.
    Used by [`download_checkpoint`][thunder.utils.download_checkpoint] and
    [`load_quartznet_checkpoint`][thunder.quartznet.compatibility.load_quartznet_checkpoint].

    Note:
        Possible values are `QuartzNet15x5Base_En`,`QuartzNet15x5Base_Zh`,`QuartzNet5x5LS_En`, `QuartzNet15x5NR_En`,
        `stt_ca_quartznet15x5`,`stt_it_quartznet15x5`,`stt_fr_quartznet15x5`,`stt_es_quartznet15x5`,
        `stt_de_quartznet15x5`,`stt_pl_quartznet15x5`,`stt_ru_quartznet15x5`,`stt_en_quartznet15x5`,
        `stt_zh_quartznet15x5`
    """
    QuartzNet15x5Base_En = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo"
    QuartzNet15x5Base_Zh = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-Zh.nemo"
    QuartzNet5x5LS_En = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet5x5LS-En.nemo"
    QuartzNet15x5NR_En = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5NR-En.nemo"

    stt_ca_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_quartznet15x5/versions/1.0.0rc1/files/stt_ca_quartznet15x5.nemo"
    stt_it_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_quartznet15x5/versions/1.0.0rc1/files/stt_it_quartznet15x5.nemo"
    stt_fr_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_quartznet15x5/versions/1.0.0rc1/files/stt_fr_quartznet15x5.nemo"
    stt_es_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_quartznet15x5/versions/1.0.0rc1/files/stt_es_quartznet15x5.nemo"
    stt_de_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_quartznet15x5/versions/1.0.0rc1/files/stt_de_quartznet15x5.nemo"
    stt_pl_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_quartznet15x5/versions/1.0.0rc1/files/stt_pl_quartznet15x5.nemo"
    stt_ru_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_quartznet15x5/versions/1.0.0rc1/files/stt_ru_quartznet15x5.nemo"
    stt_en_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/versions/1.0.0rc1/files/stt_en_quartznet15x5.nemo"
    stt_zh_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_quartznet15x5/versions/1.0.0rc1/files/stt_zh_quartznet15x5.nemo"
# fmt:on


class AugmentParams(TypedDict, total=False):
    num_cutout_masks: int
    num_time_masks: int
    num_freq_masks: int
    mask_time_width: int
    mask_freq_width: int
    dropout: float


def load_components_from_quartznet_config(
    config_path: Union[str, Path],
    augment_params: AugmentParams = None,
) -> Tuple[nn.Module, nn.Module, BatchTextTransformer]:
    """Read the important parameters from the config stored inside the .nemo
    checkpoint.

    Args:
        config_path: Path to the .yaml file, usually called model_config.yaml

    Returns:
        A tuple containing, in this order, the encoder, the audio transform and the text transform
    """
    augment_params = augment_params or {}
    conf = OmegaConf.load(config_path)
    encoder_params = conf["encoder"]["params"]
    quartznet_conf = OmegaConf.to_container(encoder_params["jasper"])

    body_config = quartznet_conf[1:-2]

    filters = [cfg["filters"] for cfg in body_config]
    kernel_sizes = [cfg["kernel"][0] for cfg in body_config]
    encoder_cfg = {
        "filters": filters,
        "kernel_sizes": kernel_sizes,
        "dropout": augment_params.pop("dropout", 0.0),
    }
    preprocess = conf["preprocessor"]["params"]

    preprocess_cfg = {
        "sample_rate": preprocess["sample_rate"],
        "n_window_size": int(preprocess["window_size"] * preprocess["sample_rate"]),
        "n_window_stride": int(preprocess["window_stride"] * preprocess["sample_rate"]),
        "n_fft": preprocess["n_fft"],
        "nfilt": preprocess["features"],
        "dither": preprocess["dither"],
        **augment_params,
    }

    labels = (
        conf["labels"] if "labels" in conf else conf["decoder"]["params"]["vocabulary"]
    )

    audio_transform = FilterbankFeatures(**preprocess_cfg)
    encoder = QuartznetEncoder(**encoder_cfg)
    text_transform = BatchTextTransformer(
        tokens=OmegaConf.to_container(labels),
    )

    return (
        encoder,
        audio_transform,
        text_transform,
    )


def load_quartznet_weights(encoder: nn.Module, decoder: nn.Module, weights_path: str):
    """Load Quartznet model weights from data present inside .nemo file

    Args:
        encoder: Encoder module to load the weights into
        decoder: Decoder module to load the weights into
        weights_path: Path to the pytorch weights checkpoint
    """
    weights = torch.load(weights_path)

    def fix_encoder_name(x: str) -> str:
        x = x.replace("encoder.", "").replace(".res.0", ".res")
        # Add another abstraction layer if it's not a masked conv
        # This is caused by the new Masked wrapper
        if ".conv" not in x:
            parts = x.split(".")
            x = ".".join(parts[:3] + ["layer", "0"] + parts[3:])
        return x

    # We remove the 'encoder.' and 'decoder.' prefix from the weights to enable
    # compatibility to load with plain nn.Modules created by reading the config
    encoder_weights = {
        fix_encoder_name(k): v for k, v in weights.items() if "encoder" in k
    }
    encoder.load_state_dict(encoder_weights, strict=True)

    decoder_weights = {
        k.replace("decoder.decoder_layers.0.", ""): v
        for k, v in weights.items()
        if "decoder" in k
    }
    decoder.load_state_dict(decoder_weights, strict=True)


def load_quartznet_checkpoint(
    checkpoint: Union[str, QuartznetCheckpoint],
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
    if isinstance(checkpoint, QuartznetCheckpoint):
        nemo_filepath = download_checkpoint(checkpoint, save_folder)
    else:
        nemo_filepath = Path(checkpoint)

    with TemporaryDirectory() as extract_folder:
        extract_archive(str(nemo_filepath), extract_folder)
        extract_path = Path(extract_folder)
        config_path = extract_path / "model_config.yaml"
        (
            encoder,
            audio_transform,
            text_transform,
        ) = load_components_from_quartznet_config(config_path, augment_params)

        decoder = conv1d_decoder(1024, text_transform.num_tokens)

        weights_path = extract_path / "model_weights.ckpt"
        load_quartznet_weights(encoder, decoder, str(weights_path))
        module = BaseCTCModule(
            encoder,
            decoder,
            audio_transform,
            text_transform,
            encoder_final_dimension=1024,
        )
        return module.eval()
