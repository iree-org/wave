"""
Transformation of tensors, such as transpose, broadcast, split, concatenate, etc.
"""

"""
GEMM Examples

Demonstrates matrix multiplication patterns including basic GEMM, dynamic expert selection,
input reordering, scatter operations, and conditional weight application.
"""

import torch
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
from wave_lang.kernel._support.indexing import sym
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config


def split_tensor_test():

    M = sym.M
    N = sym.N
    TWO_N = sym.TWO_N
    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N

    wave_size = 64

    datatype = tkl.i32

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)

    x1_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, TWO_N: k},
        outputs={M: i, N: j},
        dynamic_val_mappings={TWO_N: j},
    )
    x2_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, TWO_N: k + N},
        outputs={M: i, N: j},
        dynamic_val_mappings={TWO_N: j},
    )

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def split_tensor(
        tensor: tkl.Memory[M, TWO_N, GLOBAL_ADDRESS_SPACE, datatype],
        out1: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, datatype],
        out2: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, datatype],
    ):
        # Compute global thread ID accounting for workgroup offset
        tid = tkw.scalar(THREAD_0 + WORKGROUP_0 * wave_size, tkl.i32)
        x1_reg = tkw.read(tensor, mapping=x1_read_map, mapping_dynamic_vals=(tid,))
        x2_reg = tkw.read(tensor, mapping=x2_read_map, mapping_dynamic_vals=(tid,))
        tkw.write(x1_reg, out1)
        tkw.write(x2_reg, out2)

    hyperparams = {
        M: 64,
        N: 64,
        TWO_N: 128,
        BLOCK_M: 64,
        BLOCK_N: 64,
    }

    options = WaveCompileOptions(subs=hyperparams)
    options = set_default_run_config(options)
    split_tensor = wave_compile(options, split_tensor)

    tensor = torch.arange(64 * 128, dtype=torch.int32, device="cuda")
    tensor = tensor.view(64, 128)
    out1 = torch.zeros(64, 64, dtype=torch.int32, device="cuda")
    out2 = torch.zeros(64, 64, dtype=torch.int32, device="cuda")
    split_tensor(tensor, out1, out2)
    print("Out1: ", out1)
    print("Out2: ", out2)


if __name__ == "__main__":
    split_tensor_test()
