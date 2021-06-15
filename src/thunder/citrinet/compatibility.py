"""Helper functions to load the Citrinet model from original Nemo released checkpoint files.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["CitrinetCheckpoint", "read_params_from_config_citrinet", "fix_vocab"]

from enum import Enum
from typing import Dict, List, Tuple

from omegaconf import OmegaConf


# fmt:off
class CitrinetCheckpoint(str, Enum):
    """Trained model weight checkpoints.
    Used by [`download_checkpoint`][thunder.quartznet.compatibility.download_checkpoint] and
    [`CitrinetModule.load_from_nemo`][thunder.citrinet.module.CitrinetModule.load_from_nemo].

    Note:
        Possible values are `stt_en_citrinet_256`,`stt_en_citrinet_512`,`stt_en_citrinet_1024`, `stt_es_citrinet_512`
    """
    stt_en_citrinet_256 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256/versions/1.0.0rc1/files/stt_en_citrinet_256.nemo"
    stt_en_citrinet_512 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512/versions/1.0.0rc1/files/stt_en_citrinet_512.nemo"
    stt_en_citrinet_1024 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024/versions/1.0.0rc1/files/stt_en_citrinet_1024.nemo"
    stt_es_citrinet_512 = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_citrinet_512/versions/1.0.0/files/stt_es_citrinet_512.nemo"

    @staticmethod
    def from_string(name):
        """Creates enum value from string. Helper to use with argparse/hydra

        Args:
            name : Name of the checkpoint

        Raises:
            ValueError: Name provided is not a valid checkpoint

        Returns:
            Enum value corresponding to the name
        """
        try:
            return CitrinetCheckpoint[name]
        except KeyError as option_does_not_exist:
            raise ValueError("Name provided is not a valid checkpoint") from option_does_not_exist
# fmt:on


def read_params_from_config_citrinet(config_path: str) -> Tuple[Dict, List[str], Dict]:
    """Read the important parameters from the config stored inside the .nemo
    checkpoint.

    Args:
        config_path : Path to the .yaml file, usually called model_config.yaml

    Returns:
        A tuple containing, in this order, the encoder hyperparameters, the vocabulary, and the preprocessing hyperparameters
    """
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
    }
    preprocess = conf["preprocessor"]

    preprocess_cfg = {
        "sample_rate": preprocess["sample_rate"],
        "n_window_size": int(preprocess["window_size"] * preprocess["sample_rate"]),
        "n_window_stride": int(preprocess["window_stride"] * preprocess["sample_rate"]),
        "n_fft": preprocess["n_fft"],
        "nfilt": preprocess["features"],
        "dither": preprocess["dither"],
    }

    labels = conf["labels"] if "labels" in conf else conf["decoder"]["vocabulary"]

    return (
        encoder_cfg,
        OmegaConf.to_container(labels),
        preprocess_cfg,
    )


def fix_vocab(vocab_tokens: List[str]) -> List[str]:
    """Transform the nemo vocab tokens back to the sentencepiece sytle
    with the _ prefix

    Args:
        vocab_tokens : List of tokens in the vocabulary

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
