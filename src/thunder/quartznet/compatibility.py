"""Helper functions to load the Quartznet model from original Nemo released checkpoint files.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omegaconf import OmegaConf
from torch import nn
from torchaudio.datasets.utils import download_url

from thunder.utils import get_default_cache_folder

checkpoint_archives = {
    "QuartzNet15x5Base-En": "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo",
    "QuartzNet15x5Base-Zh": "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-Zh.nemo",
    "QuartzNet5x5LS-En": "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet5x5LS-En.nemo",
    "QuartzNet15x5NR-En": "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5NR-En.nemo",
}


def download_checkpoint(name: str, checkpoint_folder: str = None) -> Path:
    """Download quartznet checkpoint by identifier.

    Args:
        name: Model identifier. Check checkpoint_archives.keys()
        checkpoint_folder: Folder where the checkpoint will be saved to.

    Returns:
        Path to the saved checkpoint file.
    """
    if checkpoint_folder is None:
        checkpoint_folder = get_default_cache_folder()

    url = checkpoint_archives[name]
    filename = url.split("/")[-1]
    checkpoint_path = Path(checkpoint_folder) / filename
    if not checkpoint_path.exists():
        download_url(
            url,
            download_folder=checkpoint_folder,
            resume=True,
        )

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

    return (
        encoder_cfg,
        OmegaConf.to_container(conf["labels"]),
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

    # We remove the 'encoder.' and 'decoder.' prefix from the weights to enable
    # compatibility to load with plain nn.Modules created by reading the config
    encoder_weights = {
        k.replace("encoder.", "").replace(".conv", "").replace(".res.0", ".res"): v
        for k, v in weights.items()
        if "encoder" in k
    }
    encoder.load_state_dict(encoder_weights, strict=True)

    decoder_weights = {
        k.replace("decoder.decoder_layers.0.", ""): v
        for k, v in weights.items()
        if "decoder" in k
    }
    decoder.load_state_dict(decoder_weights, strict=True)
