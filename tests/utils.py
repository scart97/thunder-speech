# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

import os
from typing import List

import pytest

import torch
from pytorch_lightning import seed_everything
from torch import nn

requirescuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires cuda."
)


mark_slow = pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="Skip slow tests")


# ##############################################
# Awesome resource on deep learning testing.
# Inspired various tests here.
# https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html
# ##############################################


def _test_parameters_update(model: nn.Module, x: List[torch.Tensor]):
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    outputs = model(x)
    if isinstance(outputs, list):
        outputs = outputs[-1]
    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert (torch.sum(param.grad ** 2) != 0.0).all()


def _test_device_move(model: nn.Module, x: torch.Tensor):
    model.eval()
    seed_everything(42)
    outputs_cpu = model(x)

    seed_everything(42)
    model = model.cuda()
    outputs_gpu = model(x.cuda())

    seed_everything(42)
    model = model.cpu()
    outputs_back_on_cpu = model(x.cpu())
    model.train()
    assert torch.allclose(outputs_cpu, outputs_gpu.cpu(), atol=1e-4)
    assert torch.allclose(outputs_cpu, outputs_back_on_cpu)


def _test_batch_independence(model: nn.Module, x: torch.Tensor):
    x.requires_grad_(True)

    # Compute forward pass in eval mode to deactivate batch norm
    model.eval()
    outputs = model(x)
    model.train()

    # Mask loss for certain samples in batch
    batch_size = x.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(outputs)
    mask[mask_idx] = 0
    outputs = outputs * mask

    # Compute backward pass
    loss = outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(x.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)
