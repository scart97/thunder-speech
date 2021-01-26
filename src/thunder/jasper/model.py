"""All of the stuff to load the Jasper checkpoint
"""

# https://github.com/NVIDIA/NeMo/blob/a0fa01ab9daf453d82dee029c04af7448421005f/nemo/core/classes/modelPT.py#L1229
# https://github.com/NVIDIA/NeMo/blob/f0378b10cdadc82752d70032bd80219e531519c0/nemo/collections/asr/modules/conv_asr.py
# https://github.com/NVIDIA/NeMo/blob/a0fa01ab9daf453d82dee029c04af7448421005f/nemo/collections/asr/models/ctc_models.py#L41

from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf
from torch import nn
from torchaudio.datasets.utils import download_url, extract_archive

from thunder.jasper.blocks import JasperBlock, init_weights, jasper_activations

checkpoint_archives = {
    "QuartzNet15x5Base-En": "https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo"
}


def read_config(config_path: Path) -> Tuple[nn.Module, nn.Module]:
    """Read .yaml config and creates the encoder and decoder modules

    Args:
        config_path: Hydra config describing the Jasper/Quartznet model

    Returns:
        Encoder and decoder Modules randomly initializated
    """
    conf = OmegaConf.load(config_path)
    encoder_params = conf["encoder"]["params"]
    inplanes = encoder_params["feat_in"] * encoder_params.get("frame_splicing", 1)
    jasper_conf = OmegaConf.to_container(encoder_params["jasper"])
    activation = jasper_activations[encoder_params["activation"]]()
    residual_panes = []

    layers = []
    for cfg in jasper_conf:
        if cfg.get("residual_dense", False):
            residual_panes.append(inplanes)
        cfg["conv_mask"] = encoder_params["conv_mask"]
        cfg["planes"] = cfg.pop("filters")
        cfg["kernel_size"] = cfg.pop("kernel")
        layers.append(
            JasperBlock(
                inplanes=inplanes,
                activation=activation,
                residual_panes=residual_panes,
                **cfg
            )
        )
        inplanes = cfg["planes"]
    encoder = nn.Sequential(*layers)

    encoder.apply(init_weights)

    decoder_params = conf["decoder"]["params"]
    decoder = torch.nn.Sequential(
        torch.nn.Conv1d(
            decoder_params["feat_in"],
            decoder_params["num_classes"] + 1,
            kernel_size=1,
            bias=True,
        )
    )
    decoder.apply(init_weights)

    return encoder, decoder


def load_jasper_weights(
    config_path: str, weights_path: str
) -> Tuple[nn.Module, nn.Module]:
    """Load Jasper/Quartznet model from data present inside .nemo file

    Returns:
        Encoder and decoder Modules with the checkpoint weights loaded
    """
    encoder, decoder = read_config(config_path)

    weights = torch.load(weights_path)

    # We remove the 'encoder.' and 'decoder.' prefix from the weights to enable
    # compatibility to load with plain nn.Modules created by reading the config
    encoder_weights = {
        k.replace("encoder.", ""): v for k, v in weights.items() if "encoder" in k
    }
    encoder.load_state_dict(encoder_weights, strict=True)

    decoder_weights = {
        k.replace("decoder.decoder_layers.", ""): v
        for k, v in weights.items()
        if "decoder" in k
    }
    decoder.load_state_dict(decoder_weights, strict=True)
    return encoder, decoder


def get_jasper(name: str, checkpoint_folder: str) -> Tuple[nn.Module, nn.Module]:
    """Get Jasper model by idenfitier.
        This method downloads the checkpoint, creates the corresponding model
        and load the weights.

    Args:
        name: Model idenfitier. Check checkpoint_archives.keys()
        checkpoint_folder: Folder where the checkpoint will be saved to.

    Returns:
        Encoder and decoder Modules with the checkpoint weights loaded
    """
    url = checkpoint_archives[name]
    download_url(
        url,
        download_folder=checkpoint_folder,
        resume=True,
    )
    filename = url.split("/")[-1]
    checkpoint_path = Path(checkpoint_folder) / filename
    extract_path = Path(checkpoint_folder) / filename.replace(".nemo", "")
    extract_archive(str(checkpoint_path), extract_path)
    return load_jasper_weights(
        extract_path / "model_config.yaml", extract_path / "model_weights.ckpt"
    )


if __name__ == "__main__":
    encoder, decoder = get_jasper(
        "QuartzNet15x5Base-En", "/home/scart/audio/thunder-speech/models"
    )
    print(encoder)
