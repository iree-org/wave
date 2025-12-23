# REQUIRES: dwarfdump
# RUN: rm -rf %t && mkdir -p %t
# RUN: python %s %t
# RUN: dwarfdump --print-lines %t/*.hsaco | FileCheck %s

"""
End-to-end test for debug information in compiled Wave kernels.  This test
verifies that debug locations are preserved through the entire compilation
pipeline to DWARF.
"""

import sys
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@run_test
def test_debug_dwarf():
    # Get the output directory from command line
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "test_debug_output"

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_with_debug(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, tkl.AddressSpace.GLOBAL_MEMORY, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=False,
        # Enable debug info capture
        location_capture_config=LocationCaptureConfig(
            LocationCaptureLevel.FILE_LINE_COL
        ),
        drop_debug_info_before_mlir=False,
        # Dump the compiled binaries so we can inspect them
        dump_binaries=output_dir,
    )

    wave_compile(options, gemm_with_debug)


# Check that the line table header contains our source file
# CHECK: uri: "filepath"
# CHECK-NEXT: uri: "{{.*}}/debug_dwarf.py"

# Verify that the line table contains entries for the key operations in our Wave kernel.
# Line 61: a_reg = tkw.read(a)
# Line 62: b_reg = tkw.read(b)
# Line 63: acc = tkw.mma(a_reg, b_reg, acc)
# Line 66: tkw.write(repeat, c)
#####        Address           Line,Column Info
# CHECK-DAG: {{0x[0-9a-f]+}} [ 61,{{.*}}] NS
# CHECK-DAG: {{0x[0-9a-f]+}} [ 62,{{.*}}] NS
# CHECK-DAG: {{0x[0-9a-f]+}} [ 63,{{.*}}] NS
# CHECK-DAG: {{0x[0-9a-f]+}} [ 66,{{.*}}] NS
