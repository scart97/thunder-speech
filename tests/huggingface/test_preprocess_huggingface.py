# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

import torch
from transformers import Wav2Vec2FeatureExtractor

from thunder.huggingface.transform import Wav2Vec2Preprocess


def _get_original_results(input_tensor: torch.Tensor, return_mask: bool = False):
    original_tfm = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=return_mask,
        do_normalize=True,
    )
    original_result = [tensor.numpy() for tensor in input_tensor]
    return original_tfm(
        original_result,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )


def test_feature_extractor():
    thunder_tfm = Wav2Vec2Preprocess()

    input_tensor = torch.randn(5, 16000)
    original_result = _get_original_results(input_tensor, return_mask=False)
    lens = torch.Tensor([16000] * 5)
    thunder_result, _ = thunder_tfm(input_tensor, lens)
    assert torch.allclose(original_result.input_values, thunder_result, atol=1e-3)


def test_masked_feature_extractor():
    thunder_tfm = Wav2Vec2Preprocess(mask_input=True)

    input_tensor = torch.randn(3, 160000)
    input_tensor[1, 100000:] = 0
    input_tensor[2, 80000:] = 0
    original_lengths = torch.tensor([160000, 100000, 80000])

    original_result = _get_original_results(
        [
            input_tensor[0, :],
            input_tensor[1, :100000],
            input_tensor[2, :80000],
        ],
        return_mask=True,
    )
    thunder_result, _ = thunder_tfm(input_tensor, original_lengths)

    assert torch.allclose(original_result.input_values, thunder_result, atol=1e-3)
