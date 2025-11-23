import pytest
import torch
from torch.testing import assert_close

import wave_lang.dynamo.register_backend


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Dynamo Wave backend can compile only when torch can detect GPUs",
)
def test_wave_dynamo_backend():
    assert "wave" in torch.compiler.list_backends()

    a = torch.randn(64, 64, dtype=torch.float16)
    b = torch.randn(64, 64, dtype=torch.float16)

    def model(x, y):
        a = torch.matmul(x, y)
        return a

    # Test eager execution
    with torch.no_grad():
        expected_output = model(a, b)

    # Test compiled model execution with Wave kernel
    compiled_model = torch.compile(model, backend="wave")
    with torch.no_grad():
        actual_output = compiled_model(a, b).to(torch.float16)

    assert_close(expected_output, actual_output, check_device=False)
