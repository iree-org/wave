# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager

import torch

DEFAULT_GPU_DEVICE = None


def get_default_gpu_device_name() -> str:
    if DEFAULT_GPU_DEVICE is None:
        return "cuda"

    return f"cuda:{DEFAULT_GPU_DEVICE}"


@contextmanager
def override_default_gpu_device(device: int | None):
    """Context manager to temporarily override the default GPU device.

    Args:
        device: The GPU device index to use, or None for default "cuda".

    Example:
        with override_default_gpu_device(1):
            # All device_* functions will use cuda:1
            tensor = device_zeros(10, 10)
    """
    global DEFAULT_GPU_DEVICE
    old_device = DEFAULT_GPU_DEVICE
    DEFAULT_GPU_DEVICE = device
    try:
        yield
    finally:
        DEFAULT_GPU_DEVICE = old_device


def get_default_device() -> str:
    return (
        get_default_gpu_device_name()
        if DEFAULT_GPU_DEVICE != -1 and torch.cuda.is_available()
        else "cpu"
    )


def to_default_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(get_default_device())


def device_arange(*args, **kwargs):
    return to_default_device(torch.arange(*args, **kwargs))


def device_empty(*args, **kwargs):
    return to_default_device(torch.empty(*args, **kwargs))


def device_full(*args, **kwargs):
    return to_default_device(torch.full(*args, **kwargs))


def device_randn(*args, **kwargs):
    return to_default_device(torch.randn(*args, **kwargs))


def device_randint(*args, **kwargs):
    return to_default_device(torch.randint(*args, **kwargs))


def device_randperm(*args, **kwargs):
    return to_default_device(torch.randperm(*args, **kwargs))


def device_tensor(*args, **kwargs):
    return to_default_device(torch.tensor(*args, **kwargs))


def device_zeros(*args, **kwargs):
    return to_default_device(torch.zeros(*args, **kwargs))


def device_ones(*args, **kwargs):
    return to_default_device(torch.ones(*args, **kwargs))


def quantized_tensor(shape: tuple[int], dtype: torch.dtype, scale: float):
    return torch.ceil(
        torch.clamp(device_randn(shape, dtype=dtype) * scale, -scale, scale)
    )
