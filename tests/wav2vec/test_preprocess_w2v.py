# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

import torch
from transformers import Wav2Vec2FeatureExtractor

from thunder.wav2vec.transform import Wav2Vec2Preprocess


def test_feature_extractor():
    original_tfm = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
    )

    thunder_tfm = Wav2Vec2Preprocess()

    input_tensor = torch.randn(1, 16000)

    original_result = original_tfm(
        input_tensor[0].numpy(), sampling_rate=16000, return_tensors="pt", padding=True
    )

    thunder_result = thunder_tfm(input_tensor)
    assert torch.allclose(original_result.input_values, thunder_result, atol=1e-4)
