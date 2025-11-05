import pytest
import torch
from torch.testing import assert_close

import wave_lang.dynamo.register_backend


def test_wave_dynamo_backend():
    assert "wave" in torch.compiler.list_backends()

    a = torch.randn(64, 64)
    b = torch.randn(64, 64)

    def model(x, y):
        a = torch.matmul(x, y.T)
        return a

    # Test eager execution
    with torch.no_grad():
        expected_output = model(a, b)

    # Test compiled model execution with Wave kernel
    compiled_model = torch.compile(model, backend="wave")
    with torch.no_grad():
        actual_output = compiled_model(a, b)

    assert_close(expected_output, actual_output, check_device=False)
