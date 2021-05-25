"""Helper functions to load the Quartznet model from original Nemo released checkpoint files.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = [
    "NemoCheckpoint",
    "download_checkpoint",
    "read_params_from_config",
    "load_quartznet_weights",
]

from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wget
from omegaconf import OmegaConf
from torch import nn

from thunder.utils import get_default_cache_folder


# fmt:off
class NemoCheckpoint(str, Enum):
    """Trained model weight checkpoints.
    Used by [`download_checkpoint`][thunder.quartznet.compatibility.download_checkpoint] and
    [`QuartznetModule.load_from_nemo`][thunder.quartznet.module.QuartznetModule.load_from_nemo].

    Note:
        Possible values are `QuartzNet15x5Base_En`,`QuartzNet15x5Base_Zh`,`QuartzNet5x5LS_En`, `QuartzNet15x5NR_En`,
        `stt_ca_quartznet15x5`,`stt_it_quartznet15x5`,`stt_fr_quartznet15x5`,`stt_es_quartznet15x5`,
        `stt_de_quartznet15x5`,`stt_pl_quartznet15x5`,`stt_ru_quartznet15x5`,`stt_en_quartznet15x5`,
        `stt_zh_quartznet15x5`
    """
    QuartzNet15x5Base_En = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo"
    QuartzNet15x5Base_Zh = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-Zh.nemo",
    QuartzNet5x5LS_En = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet5x5LS-En.nemo",
    QuartzNet15x5NR_En = "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5NR-En.nemo",

    stt_ca_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_quartznet15x5/versions/1.0.0rc1/files/stt_ca_quartznet15x5.nemo",
    stt_it_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_quartznet15x5/versions/1.0.0rc1/files/stt_it_quartznet15x5.nemo",
    stt_fr_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_quartznet15x5/versions/1.0.0rc1/files/stt_fr_quartznet15x5.nemo",
    stt_es_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_quartznet15x5/versions/1.0.0rc1/files/stt_es_quartznet15x5.nemo",
    stt_de_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_quartznet15x5/versions/1.0.0rc1/files/stt_de_quartznet15x5.nemo",
    stt_pl_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_quartznet15x5/versions/1.0.0rc1/files/stt_pl_quartznet15x5.nemo",
    stt_ru_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_quartznet15x5/versions/1.0.0rc1/files/stt_ru_quartznet15x5.nemo",
    stt_en_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/versions/1.0.0rc1/files/stt_en_quartznet15x5.nemo",
    stt_zh_quartznet15x5 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_quartznet15x5/versions/1.0.0rc1/files/stt_zh_quartznet15x5.nemo",

    @staticmethod
    def from_string(name):
        try:
            return NemoCheckpoint[name]
        except KeyError:
            raise ValueError()
# fmt:on


def download_checkpoint(name: NemoCheckpoint, checkpoint_folder: str = None) -> Path:
    """Download quartznet checkpoint by identifier.

    Args:
        name: Model identifier. Check checkpoint_archives.keys()
        checkpoint_folder: Folder where the checkpoint will be saved to.

    Returns:
        Path to the saved checkpoint file.
    """
    if checkpoint_folder is None:
        checkpoint_folder = get_default_cache_folder()

    url = name.value
    filename = url.split("/")[-1]
    checkpoint_path = Path(checkpoint_folder) / filename
    if not checkpoint_path.exists():
        wget.download(url, out=str(checkpoint_path))

    return checkpoint_path


def read_params_from_config(config_path: str) -> Tuple[Dict, List[str], Dict]:
    """Read the important parameters from the config stored inside the .nemo
    checkpoint.

    Args:
        config_path : Path to the .yaml file, usually called model_config.yaml

    Returns:
        A tuple containing, in this order, the encoder hyperparameters, the vocabulary, and the preprocessing hyperparameters
    """
    conf = OmegaConf.load(config_path)
    encoder_params = conf["encoder"]["params"]
    quartznet_conf = OmegaConf.to_container(encoder_params["jasper"])

    body_config = quartznet_conf[1:-2]

    filters = [cfg["filters"] for cfg in body_config]
    kernel_sizes = [cfg["kernel"][0] for cfg in body_config]
    encoder_cfg = {
        "filters": filters,
        "kernel_sizes": kernel_sizes,
    }
    preprocess = conf["preprocessor"]["params"]

    preprocess_cfg = {
        "sample_rate": preprocess["sample_rate"],
        "n_window_size": int(preprocess["window_size"] * preprocess["sample_rate"]),
        "n_window_stride": int(preprocess["window_stride"] * preprocess["sample_rate"]),
        "n_fft": preprocess["n_fft"],
        "nfilt": preprocess["features"],
        "dither": preprocess["dither"],
    }

    labels = (
        conf["labels"] if "labels" in conf else conf["decoder"]["params"]["vocabulary"]
    )

    return (
        encoder_cfg,
        OmegaConf.to_container(labels),
        preprocess_cfg,
    )


def load_quartznet_weights(encoder: nn.Module, decoder: nn.Module, weights_path: str):
    """Load Quartznet model weights from data present inside .nemo file

    Args:
        encoder: Encoder module to load the weights into
        decoder: Decoder module to load the weights into
        weights_path : Path to the pytorch weights checkpoint
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
