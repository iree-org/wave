"""
MXFP4 Scaled GEMM for GFX950 (MI350) - No Manual Schedule

Usage:
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm --debug
    python 7_mxfp_gemm.py --test test_basic_mxfp_gemm --shape 512,512,4096
    python 7_mxfp_gemm.py --list_tests
"""

import torch

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
)

from utils import parse_args, list_tests, run_test


def test_basic_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """MXFP4 scaled GEMM with compiler-default scheduling (no manual schedule)."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=8)

    # Override to use default scheduling instead of manual
    options.schedule = None
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    if is_debug:
        with open("gemm_mxfp4_basic.mlir", "w") as f:
            f.write(compiled_gemm.asm)
        print("MLIR written to gemm_mxfp4_basic.mlir")

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    compiled_gemm(x, x_scales, w.T.contiguous(), w_scales, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )
    print("Basic MXFP GEMM test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test, globals(), args.debug, args.repeat, args.shape, args.block
    )
    exit(0 if success else 1)
