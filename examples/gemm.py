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
    return parser.parse_args()


def list_tests():
    # find all the functions in the file that end with _test
    tests = [f for f in globals() if f.endswith("_test")]
    print("Available tests:")
    for test in tests:
        print(f"  {test}")


def simple_gemm_test():
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64, mma_type=tkw.MMAType.F32_16x16x16_F16
        ),
    ]

    @tkw.wave(constraints)
    def gemm(
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
    m, n, k = 64, 64, 128  # Small dimensions for testing

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
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
    }

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    # Run the GEMM kernel
    compiled_gemm(a, b, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b.t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def downcast_gemm_test():
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
    def gemm(
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
    compiled_gemm = wave_compile(options, gemm)

    # Run the GEMM kernel
    compiled_gemm(a, b, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

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
    def gemm(
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
    compiled_gemm = wave_compile(options, gemm)

    # Run the GEMM kernel
    compiled_gemm(a, b, 1, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def dyn_downcast_gemm_test():
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
    def gemm(
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
        print_ir_after="all",
        print_ir_before="all",
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    # Run the GEMM kernel
    compiled_gemm(a, b, 1, c)
    print(compiled_gemm.asm)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b[1].t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def reorder_a_gemm_test():
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
    def gemm(
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
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)
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


def scatter_gemm_test():
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
            vector_shapes={E: E, M_DIV_2: 16, M: 16, K: K},
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
    def gemm(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[M_DIV_2, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        idx: i32,
    ):
        # Initialize the accumulator register with zeros
        c_reg = Register[M, N, f32](0.0)
        tkw.set_symbol(IDX, idx)
        a_mock = tkw.read(a_back)

        @tkw.conditional(THREAD_0 < M_DIV_2)
        def scatter_op():
            tid = tkw.scalar(THREAD_0, i32)
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid,),
                elements_per_thread=1,
            )

            @tkw.iterate(K, init_args=[])
            def copy_row():
                a_row_data = tkw.read(
                    a,
                    mapping=a_read_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=4,
                )
                tkw.write(
                    a_row_data,
                    a_back,
                    mapping=a_write_map,
                    mapping_dynamic_vals=(tid,),
                    elements_per_thread=4,
                )

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
    options = WaveCompileOptions(
        subs=hyperparams,
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    with open("scatter_gemm.mlir", "w") as f:
        f.write(compiled_gemm.asm)

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

    breakpoint()
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


if __name__ == "__main__":
    args = parse_args()
    if args.list_tests:
        list_tests()
    else:
        globals()[args.test]()
