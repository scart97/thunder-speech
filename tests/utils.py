# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

import os

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


def call_model(model, x, device="cpu"):
    if isinstance(x, tuple):
        x = tuple(i.to(device) for i in x)
        out, _ = model(*x)
    else:
        out = model(x.to(device))
    return out


def _test_parameters_update(model: nn.Module, x):
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    outputs = call_model(model, x)

    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert (torch.sum(param.grad ** 2) != 0.0).all()


def _test_device_move(model: nn.Module, x: torch.Tensor, atol: float = 1e-4):
    model.eval()
    seed_everything(42)
    outputs_cpu = call_model(model, x)

    seed_everything(42)
    model = model.cuda()
    outputs_gpu = call_model(model, x, "cuda")

    seed_everything(42)
    model = model.cpu()
    outputs_back_on_cpu = call_model(model, x)
    model.train()
    assert torch.allclose(outputs_cpu, outputs_gpu.cpu(), atol=atol)
    assert torch.allclose(outputs_cpu, outputs_back_on_cpu)


def _test_batch_independence(model: nn.Module, inp: torch.Tensor):
    if isinstance(inp, tuple):
        x = inp[0]
    else:
        x = inp
    x.requires_grad_(True)

    # Compute forward pass in eval mode to deactivate batch norm
    model.eval()
    outputs = call_model(model, inp)
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
