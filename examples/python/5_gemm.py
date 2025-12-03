"""
GEMM Examples

Demonstrates matrix multiplication patterns including basic GEMM, dynamic expert selection,
input reordering, scatter operations, and conditional weight application.
"""

import torch
import argparse

import wave_lang.kernel.wave as tkw
from wave_lang.kernel._support.dtype import f16, f32, i32
from wave_lang.kernel._support.indexing import sym
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

# Define symbolic dimensions for our matrices
M = sym.M  # Rows of A and C
N = sym.N  # Rows of B and columns of C
K = sym.K  # Columns of A and B

# Define workgroup tile sizes
BLOCK_M = sym.BLOCK_M
BLOCK_N = sym.BLOCK_N
BLOCK_K = sym.BLOCK_K

# Define the address space for our memory buffers
ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C


def parse_args():
    parser = argparse.ArgumentParser()
    # one of the tests or list_tests is required
    parser.add_argument("--test", type=str, required=False)
    parser.add_argument("--list_tests", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def list_tests():
    # find all the functions in the file that end with _test
    tests = [f for f in globals() if f.endswith("_test")]
    print("Available tests:")
    for test in tests:
        print(f"  {test}")


def simple_gemm_test(is_debug=False):
    """Basic matrix multiplication kernel."""
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        ),
    ]

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        c_reg = Register[M, N, f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

    # Create test matrices
    m, n, k = 64, 64, 64  # Small dimensions for testing

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
    }

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        print_ir_after="all" if is_debug else [],
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    if is_debug:
        print(compiled_gemm.asm)

    # Run the GEMM kernel
    compiled_gemm(a, b, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b.t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def test_gemm_select_expert(is_debug=False):
    """GEMM with expert selection from 3D tensor using fixed and dynamic indexing."""
    E = sym.E
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={E: E},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: sympy.Integer(1), N: i, K: j},
        outputs={N: i, K: j},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        c_reg = Register[M, N, f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a)
            b_reg = tkw.read(b, mapping=mapping)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

    # Create test matrices
    m, n, k = 64, 64, 128  # Small dimensions for testing
    e = 8

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(e, n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_B: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
    }

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    # Run the GEMM kernel
    compiled_gemm(a, b, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    if is_debug:
        print(compiled_gemm.asm)
    print("GEMM test passed!")

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    IDX = sym.IDX
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        idx: i32,
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        c_reg = Register[M, N, f32](0.0)
        tkw.set_symbol(IDX, idx)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a)
            b_reg = tkw.read(b, mapping=mapping)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

    # Create test matrices
    m, n, k = 64, 64, 128  # Small dimensions for testing
    e = 8

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(e, n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
    }

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    # Run the GEMM kernel
    compiled_gemm(a, b, 1, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def test_gemm_dynamic_expert(is_debug=False):
    """GEMM with runtime expert selection from 3D weight tensor."""
    E = sym.E
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(E, E, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={E: E},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    IDX = sym.IDX
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={M: i, K: j},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        idx: i32,
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        c_reg = Register[M, N, f32](0.0)
        tkw.set_symbol(IDX, idx)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a, mapping=a_read_map)
            b_reg = tkw.read(b, mapping=mapping)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

    # Create test matrices
    m, n, k = 64, 64, 128  # Small dimensions for testing
    e = 8

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(e, n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
    }

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        print_ir_after="all" if is_debug else [],
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    # Run the GEMM kernel
    compiled_gemm(a, b, 1, c)
    if is_debug:
        print(compiled_gemm.asm)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def test_gemm_reordered_input(is_debug=False):
    """GEMM with dynamically reordered input rows."""
    E = sym.E
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(E, E, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={E: E, M: 16, K: 16},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    IDX = sym.IDX
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[M, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        idx: i32,
    ):
        # Initialize the accumulator register with zeros
        c_reg = Register[M, N, f32](0.0)
        a_reg = Register[M, K, f16](0.0)
        tkw.set_symbol(IDX, idx)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            reordered_idx = tkw.read(reorder_a, elements_per_thread=1)
            a_reg = tkw.read(
                a, mapping=a_read_map, mapping_dynamic_vals=(reordered_idx,)
            )
            b_reg = tkw.read(b, mapping=mapping)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)

            tkw.write(a_reg, a_back)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

    # Create test matrices
    m, n, k = 64, 64, 128  # Small dimensions for testing
    e = 8

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    a_back = torch.zeros(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(e, n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # create reorder_a such that it is a permutation of the rows of a
    reorder_a = torch.randperm(m).to(torch.int32).to(device="cuda")
    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
    }

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        print_ir_after="all" if is_debug else [],
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)
    if is_debug:
        print(compiled_gemm.asm)

    # Run the GEMM kernel
    compiled_gemm(a, b, reorder_a, a_back, c, 1)
    reordered_a = a[reorder_a]

    print("Reorder idx: ", reorder_a)
    print("A back: ", a_back[0])
    print("A: ", a[reorder_a[0]])
    print("Reordered A: ", reordered_a[0])

    assert torch.allclose(
        a_back, reordered_a, rtol=1e-2, atol=1e-2
    ), f"A back doesn't match expected output\nMax difference: {(a_back - reordered_a).abs().max()}"

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(reordered_a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def test_gather_rows(is_debug=False):
    """Gather operation to reorder matrix rows based on dynamic indices."""
    M_DIV_2 = sym.M_DIV_2
    I = sym.I
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M_DIV_2: M_DIV_2, M: M, K: BLOCK_K},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    d0 = tkw.IndexMapping.dynamic_val(0)

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    a_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={M: i, K: j},
    )

    dyn_reorder_a_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={M_DIV_2: d0},
        outputs={M_DIV_2: i},
        dynamic_val_mappings={M_DIV_2: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        reorder_a: Memory[M_DIV_2, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
    ):
        # Initialize the accumulator register with zeros
        @tkw.conditional(THREAD_0 < M_DIV_2)
        def gather_op():
            tid = tkw.scalar(THREAD_0, i32)
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid,),
            )

            @tkw.iterate(K, init_args=[])
            def copy_row():
                a_row_data = tkw.read(
                    a,
                    mapping=a_read_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=BLOCK_K,
                )
                tkw.write(
                    a_row_data,
                    a_back,
                    mapping=a_write_map,
                    elements_per_thread=BLOCK_K,
                )

    # Create test matrices
    m, k = 64, 128  # Small dimensions for testing

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    a_back = torch.zeros(m, k, dtype=torch.float16, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_K: 32,
        M: m,
        K: k,
        M_DIV_2: m // 2,
    }

    # Compile the kernel
    if is_debug:
        options = WaveCompileOptions(
            subs=hyperparams,
            print_ir_after="all" if is_debug else [],
        )
    else:
        options = WaveCompileOptions(
            subs=hyperparams,
        )

    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    if is_debug:
        print(compiled_gemm.asm)

    # create reorder_a such that it is a permutation of the rows of a
    reorder_a = torch.randperm(m // 2).to(torch.int32).to(device="cuda")
    reorder_a_clone = reorder_a.clone().to(device="cuda")
    compiled_gemm(a, reorder_a_clone, a_back)
    reordered_a = torch.zeros((m, k), dtype=torch.float16).to(device="cuda")

    # read rows of a in reorder_a order
    for i in range(m // 2):
        reordered_a[i] = a[reorder_a[i]]

    print("Reorder idx: ", reorder_a)
    print("A back: ", a_back[0])
    print("A: ", a[reorder_a[0]])
    print("Reordered A: ", reordered_a[0])

    print("gather_a test passed!")


def test_gather_then_gemm(is_debug=False):
    """Gather rows into shared memory then perform GEMM on reordered data."""
    M_DIV_2 = sym.M_DIV_2
    SHARED_MEM = sym.SHARED_MEM
    I = sym.I
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.TilingConstraint(I),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M_DIV_2: M_DIV_2, M: 16, N: 16, K: 16, I: 0},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)
    d1 = tkw.IndexMapping.dynamic_val(1)

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    a_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    dyn_reorder_a_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={M_DIV_2: d0},
        outputs={M_DIV_2: i},
        dynamic_val_mappings={M_DIV_2: i},
    )

    a_simple_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={M: i, K: j},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[M_DIV_2, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):

        zero_reg = tkw.Register[M, K, f16](0.0)
        tkw.write(zero_reg, a_back)

        valid_threads = THREAD_0 < M_DIV_2

        @tkw.conditional(valid_threads)
        def gather_op():
            tid = tkw.Register[M_DIV_2, i32](THREAD_0)
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid,),
            )

            @tkw.iterate(K, init_args=[])
            def copy_row():
                a_row_data = tkw.read(
                    a,
                    mapping=a_read_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=16,
                )

                tkw.write(
                    a_row_data,
                    a_back,
                    mapping=a_write_map,
                    mapping_dynamic_vals=(tid,),
                    elements_per_thread=16,
                )

        tkw.workgroup_barrier()
        c_reg = Register[M, N, f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a_back)
            b_reg = tkw.read(b)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

    # Create test matrices
    m, n, k = 128, 128, 128

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    a_back = torch.zeros(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        SHARED_MEM: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        M_DIV_2: 4,
    }

    # Compile the kernel
    if is_debug:
        options = WaveCompileOptions(
            subs=hyperparams,
            print_ir_after="all" if is_debug else [],
        )
    else:
        options = WaveCompileOptions(
            subs=hyperparams,
        )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    if is_debug:
        print(compiled_gemm.asm)

    # create reorder_a such that it is a permutation of the rows of a
    reorder_a = torch.randperm(4).to(torch.int32).to(device="cuda")
    reorder_a_clone = reorder_a.clone().to(device="cuda")
    compiled_gemm(a, b, reorder_a_clone, a_back, c)
    reordered_a = torch.zeros((m, k), dtype=torch.float16).to(device="cuda")

    # read rows of a in reorder_a order
    for i in range(4):
        reordered_a[i] = a[reorder_a[i]]

    # print("Reorder idx: ", reorder_a)
    # print("A back: ", a_back[0])
    # print("A: ", a[reorder_a[0]])
    # print("Reordered A: ", reordered_a[0])

    assert torch.allclose(
        a_back, reordered_a, rtol=1e-2, atol=1e-2
    ), f"A back doesn't match expected output\nMax difference: {(a_back - reordered_a).abs().max()}"

    expected = torch.matmul(reordered_a, b.t())

    print("Expected: ", expected[0])
    print("C: ", c[0])

    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("gather_then_gemm test passed!")


def test_gather_gemm_expert(is_debug=False):
    """Gather rows, then GEMM with expert selection."""
    E = sym.E
    M_DIV_2 = sym.M_DIV_2
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(E, E, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={E: E, M_DIV_2: M_DIV_2, M: 16, N: 16, K: 16},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    IDX = sym.IDX
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    a_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    dyn_reorder_a_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={M_DIV_2: d0},
        outputs={M_DIV_2: i},
        dynamic_val_mappings={M_DIV_2: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[M_DIV_2, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        idx: i32,
    ):
        # Initialize the accumulator register with zeros
        zero_reg = Register[M, K, f16](0.0)
        tkw.write(zero_reg, a_back)

        tkw.set_symbol(IDX, idx)

        condition = THREAD_0 < M_DIV_2

        @tkw.conditional(condition)
        def gather_op():
            tid = tkw.Register[M_DIV_2, i32](THREAD_0)
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid,),
            )

            @tkw.iterate(K, init_args=[])
            def copy_row():
                a_row_data = tkw.read(
                    a,
                    mapping=a_read_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=16,
                )
                tkw.write(
                    a_row_data,
                    a_back,
                    mapping=a_write_map,
                    mapping_dynamic_vals=(tid,),
                    elements_per_thread=16,
                )

        tkw.workgroup_barrier()
        c_reg = Register[M, N, f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def gemm_compute(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a_back)
            b_reg = tkw.read(b, mapping=mapping)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(gemm_compute, c)

    # Create test matrices
    m, n, k = 64, 64, 128  # Small dimensions for testing
    e = 8

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    a_back = torch.zeros(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(e, n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
        M_DIV_2: m // 2,
    }

    # Compile the kernel
    if is_debug:
        options = WaveCompileOptions(
            subs=hyperparams,
            print_ir_after="all" if is_debug else [],
        )
    else:
        options = WaveCompileOptions(
            subs=hyperparams,
        )

    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    if is_debug:
        print(compiled_gemm.asm)

    # create reorder_a such that it is a permutation of the rows of a
    reorder_a = torch.randperm(m // 2).to(torch.int32).to(device="cuda")
    compiled_gemm(a, b, reorder_a, a_back, c, 1)
    reordered_a = torch.zeros((m, k), dtype=torch.float16).to(device="cuda")

    # read rows of a in reorder_a order
    for i in range(m // 2):
        reordered_a[i] = a[reorder_a[i]]

    print("Reorder idx: ", reorder_a)
    print("A back: ", a_back[0])
    print("A: ", a[reorder_a[0]])
    print("Reordered A: ", reordered_a[0])

    assert torch.allclose(
        a_back, reordered_a, rtol=1e-2, atol=1e-2
    ), f"A back doesn't match expected output\nMax difference: {(a_back - reordered_a).abs().max()}"

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(reordered_a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def test_gather_gemm_padded(is_debug=False):
    """Gather with padding support, GEMM, then scatter output back."""
    E = sym.E
    BLOCK_SHAPE = sym.BLOCK_SHAPE
    PAD_VALUE = sym.PAD_VALUE

    IDX = sym.IDX
    GATHER_IDX = sym.GATHER_IDX
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(E, E, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={E: E, BLOCK_SHAPE: BLOCK_SHAPE, M: 16, N: 16, K: 16},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    a_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    c_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i},
    )

    c_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: d0, N: j},
        dynamic_val_mappings={M: i},
    )

    dyn_reorder_a_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={BLOCK_SHAPE: d0},
        outputs={BLOCK_SHAPE: i},
        dynamic_val_mappings={BLOCK_SHAPE: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[BLOCK_SHAPE, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        c_back: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        idx: i32,
    ):
        # Initialize the accumulator register with zeros
        zero_reg = Register[M, K, f16](0.0)
        zero_reg_mn = Register[M, N, f32](0.0)
        tkw.write(zero_reg, a_back)
        tkw.write(zero_reg_mn, c_back)
        mock_reg = tkw.read(reorder_a)

        tkw.set_symbol(IDX, idx)

        condition = THREAD_0 < BLOCK_SHAPE

        @tkw.conditional(condition)
        def gather_op():
            tid = tkw.Register[BLOCK_SHAPE, i32](THREAD_0)
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid,),
            )

            tkw.set_symbol(GATHER_IDX, reordered_idx)
            is_not_padding = reordered_idx < tkw.scalar(PAD_VALUE, i32)

            @tkw.conditional(is_not_padding)
            def then():
                @tkw.iterate(K, init_args=[])
                def copy_row():
                    a_row_data = tkw.read(
                        a,
                        mapping=a_read_map,
                        mapping_dynamic_vals=(reordered_idx,),
                        elements_per_thread=16,
                    )
                    tkw.write(
                        a_row_data,
                        a_back,
                        mapping=a_write_map,
                        mapping_dynamic_vals=(tid,),
                        elements_per_thread=16,
                    )

        tkw.workgroup_barrier()
        c_reg = Register[M, N, f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def gemm_compute(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a_back)
            b_reg = tkw.read(b, mapping=mapping)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(gemm_compute, c_back)

        @tkw.conditional(condition)
        def scatter_op():
            tid = tkw.Register[BLOCK_SHAPE, i32](THREAD_0)
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid,),
            )

            tkw.set_symbol(GATHER_IDX, reordered_idx)
            is_not_padding = GATHER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def then():
                c_row_data = tkw.read(
                    c_back,
                    mapping=c_read_map,
                    mapping_dynamic_vals=(tid,),
                    elements_per_thread=16,
                )
                tkw.write(
                    c_row_data,
                    c,
                    mapping=c_write_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=16,
                )

    # Create test matrices
    m, n, k = 64, 64, 128  # Small dimensions for testing
    e = 8
    block_shape = 16

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    a_back = torch.zeros(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(e, n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    c_back = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
        BLOCK_SHAPE: block_shape,
        PAD_VALUE: m,
    }

    # Compile the kernel
    if is_debug:
        options = WaveCompileOptions(
            subs=hyperparams,
            print_ir_after="all" if is_debug else [],
        )
    else:
        options = WaveCompileOptions(
            subs=hyperparams,
        )

    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    if is_debug:
        print(compiled_gemm.asm)

    # create reorder_a such that it is a permutation of the rows of a
    reorder_a = torch.randperm(m).to(torch.int32).to(device="cuda")
    reorder_a = reorder_a[:block_shape]
    # make last two values of reorder_a m
    reorder_a[-2] = m
    reorder_a[-1] = m

    compiled_gemm(a, b, reorder_a, a_back, c, c_back, 1)
    reordered_a = torch.zeros((m, k), dtype=torch.float16).to(device="cuda")

    # read rows of a in reorder_a order
    for i in range(block_shape):
        if reorder_a[i] < m:
            reordered_a[i] = a[reorder_a[i]]

    print("Reorder idx: ", reorder_a)
    print("A back: ", a_back[0])
    print("A: ", a[reorder_a[0]])
    print("Reordered A: ", reordered_a[0])

    assert torch.allclose(
        a_back, reordered_a, rtol=1e-2, atol=1e-2
    ), f"A back doesn't match expected output\nMax difference: {(a_back - reordered_a).abs().max()}"

    # Verify the result using PyTorch's matmul
    expected_int = torch.matmul(reordered_a, b[1].t())
    expected = torch.zeros((m, n), dtype=torch.float32).to(device="cuda")

    for i in range(block_shape):
        if reorder_a[i] < m:
            expected[reorder_a[i]] = expected_int[i]

    assert torch.allclose(
        c_back.to(torch.float16), expected_int, rtol=1e-2, atol=1e-2
    ), f"C back doesn't match expected output\nMax difference: {(c_back.to(torch.float16) - expected_int).abs().max()}"

    assert torch.allclose(
        c, expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def test_gather_gemm_fused(is_debug=False):
    """Fused gather-GEMM-scatter with multiple experts (MOE-style)."""
    E = sym.E
    TOTAL_ELEMS = sym.TOTAL_ELEMS
    NUM_BLOCKS = sym.NUM_BLOCKS
    BLOCK_SHAPE = sym.BLOCK_SHAPE
    PAD_VALUE = sym.PAD_VALUE

    IDX = sym.IDX
    GATHER_IDX = sym.GATHER_IDX
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(TOTAL_ELEMS, BLOCK_SHAPE, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.WaveConstraint(TOTAL_ELEMS, BLOCK_SHAPE),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={
                E: E,
                TOTAL_ELEMS: TOTAL_ELEMS,
                BLOCK_SHAPE: BLOCK_SHAPE,
                M: 16,
                N: 16,
                K: 16,
                NUM_BLOCKS: NUM_BLOCKS,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    b_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    a_back_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={NUM_BLOCKS: WORKGROUP_2, M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    a_back_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: i, K: j},
        outputs={M: i, K: j},
    )

    c_back_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={NUM_BLOCKS: WORKGROUP_2, M: i, N: j},
    )

    c_back_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: d0, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i},
    )

    c_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: d0, N: j},
        dynamic_val_mappings={M: i},
    )

    dyn_reorder_a_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={TOTAL_ELEMS: d0},
        outputs={TOTAL_ELEMS: i},
        dynamic_val_mappings={TOTAL_ELEMS: i},
    )

    expert_id_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_BLOCKS: d0},
        outputs={NUM_BLOCKS: i},
        dynamic_val_mappings={NUM_BLOCKS: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[TOTAL_ELEMS, ADDRESS_SPACE_A, i32],  # Input matrix A
        expert_ids: Memory[NUM_BLOCKS, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[NUM_BLOCKS, M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        c_back: Memory[NUM_BLOCKS, M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        zero_reg = Register[M, K, f16](0.0)
        zero_reg_mn = Register[M, N, f32](0.0)
        tkw.write(zero_reg, a_back)
        tkw.write(zero_reg_mn, c_back)
        mock_reg = tkw.read(reorder_a)

        wid = tkw.scalar(WORKGROUP_2, i32)
        expert_id = tkw.read(
            expert_ids, mapping=expert_id_read_map, mapping_dynamic_vals=(wid,)
        )
        tkw.set_symbol(IDX, expert_id)
        condition = THREAD_0 < BLOCK_SHAPE

        @tkw.conditional(condition)
        def gather_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_2)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(GATHER_IDX, reordered_idx)
            is_not_padding = GATHER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def then():
                @tkw.iterate(K, init_args=[])
                def copy_row():
                    a_row_data = tkw.read(
                        a,
                        mapping=a_read_map,
                        mapping_dynamic_vals=(reordered_idx,),
                        elements_per_thread=16,
                    )

                    tkw.write(
                        a_row_data,
                        a_back,
                        mapping=a_back_write_map,
                        mapping_dynamic_vals=(tid,),
                        elements_per_thread=16,
                    )

        tkw.workgroup_barrier()
        c_reg = Register[M, N, f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def gemm_compute(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a_back, mapping=a_back_read_map)
            b_reg = tkw.read(b, mapping=b_read_map)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(gemm_compute, c_back, mapping=c_back_write_map)

        @tkw.conditional(condition)
        def scatter_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_2)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(GATHER_IDX, reordered_idx)
            is_not_padding = GATHER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def then():
                c_row_data = tkw.read(
                    c_back,
                    mapping=c_back_read_map,
                    mapping_dynamic_vals=(tid,),
                    elements_per_thread=16,
                )
                tkw.write(
                    c_row_data,
                    c,
                    mapping=c_write_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=16,
                )

    # Create test matrices
    m, n, k = 64, 64, 128  # Small dimensions for testing
    block_shape = 6
    total_elems = 30
    num_blocks = total_elems // block_shape
    num_experts = 4

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    a_back = torch.zeros(num_blocks, m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(num_experts, n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    c_back = torch.zeros(num_blocks, m, n, dtype=torch.float32, device="cuda")

    # create an expert_id list which is num_blocks long, each element is a random integer between 0 and num_experts - 1
    expert_ids = torch.randint(
        0, num_experts, (num_blocks,), dtype=torch.int32, device="cuda"
    )

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: num_experts,
        BLOCK_SHAPE: block_shape,
        TOTAL_ELEMS: total_elems,
        NUM_BLOCKS: num_blocks,
        PAD_VALUE: m,
    }

    # Compile the kernel
    if is_debug:
        options = WaveCompileOptions(
            subs=hyperparams,
            print_ir_after="all" if is_debug else [],
        )
    else:
        options = WaveCompileOptions(
            subs=hyperparams,
        )

    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    if is_debug:
        print(compiled_gemm.asm)

    # create reorder_a such that it is a permutation of the rows of a
    reorder_a = torch.randperm(total_elems).to(torch.int32).to(device="cuda")

    # make reorder 2d where each row is total_elemns/block_shape, block_shape times
    reorder_a = reorder_a.view(total_elems // block_shape, block_shape)

    # for each row, make last 0, 1, or 2 elements m randomly
    for i in range(total_elems // block_shape):
        reorder_a[i, -2] = m
        reorder_a[i, -1] = m

    reorder_a = reorder_a.view(-1)

    compiled_gemm(a, b, reorder_a, expert_ids, a_back, c, c_back)
    reordered_a = torch.zeros((num_blocks, m, k), dtype=torch.float16).to(device="cuda")

    # Verify the result using PyTorch's matmul
    expected = torch.zeros((m, n), dtype=torch.float32).to(device="cuda")
    expected_int = torch.zeros((num_blocks, m, n), dtype=torch.float32).to(
        device="cuda"
    )

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        for i in range(block_shape):
            idx = block_idx * block_shape + i
            if reorder_a[idx] < m:
                reordered_a[block_idx][i] = a[reorder_a[idx]]

        expected_int[block_idx] = torch.matmul(reordered_a[block_idx], b[expert_id].t())

        for i in range(block_shape):
            idx = block_idx * block_shape + i
            if reorder_a[idx] < m:
                expected[reorder_a[idx]] = expected_int[block_idx][i]

    # print exactly which indices are not matching
    # for i in range(num_blocks):
    #     for j in range(m):
    #         for k in range(k):
    #             if not torch.allclose(a_back[i][j][k], reordered_a[i][j][k], rtol=1e-2, atol=1e-2):
    #                 print(f"A back doesn't match expected output at index {i}, {j}, {k}")

    # for i in range(m):
    #     for j in range(n):
    #         if not torch.allclose(expected[i][j], c[i][j], rtol=1e-2, atol=1e-2):
    #             print(f"C doesn't match expected output at index {i}, {j}")

    # for i in range(num_blocks):
    #     print("EXPERT ID: ", i)
    #     try:
    #         assert torch.allclose(
    #             a_back[i], reordered_a[i], rtol=1e-2, atol=1e-2
    #         ), f"A back doesn't match expected output\nMax difference: {(a_back[i] - reordered_a[i]).abs().max()}"
    #     except Exception as e:
    #         breakpoint()
    # assert torch.allclose(
    #     a_back, reordered_a, rtol=1e-2, atol=1e-2
    # ), f"A back doesn't match expected output\nMax difference: {(a_back - reordered_a).abs().max()}"

    # assert torch.allclose(
    #     c_back.to(torch.float16), expected_int, rtol=1e-2, atol=1e-2
    # ), f"C back doesn't match expected output\nMax difference: {(c_back.to(torch.float16) - expected_int).abs().max()}"

    assert torch.allclose(
        c, expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


if __name__ == "__main__":
    args = parse_args()
    if args.list_tests:
        list_tests()
    else:
        # run the test 10 times and collect how many times it passes
        for i in range(args.repeat):
            try:
                globals()[args.test](args.debug)
                print(f"Test {i} passed")
            except Exception as e:
                print(f"Error: {e}")
                print(f"Test {i} failed")
                exit(1)
