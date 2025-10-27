"""
Atomic Operations Examples

Demonstrates atomic memory operations including atomic add with return values
and reading back scalar values after atomic operations.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
import torch

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from utils import parse_args, list_tests, run_test


M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def test_atomic_add_return_value(is_debug=False):
    """Atomic add operation that returns the old value before the addition."""
    LIMIT_VAL = tkl.sym.LIMIT_VAL

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 0, M: 64, LIMIT_VAL: 0},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    # B is iterated over and so we define a tiling constraint on it.
    # However, there is no notion of tile size for the iteration as
    # it is an unstructured loop.
    constraints += [tkw.TilingConstraint(B)]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    simple_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: i},
        outputs={M: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, ADDRESS_SPACE_0, tkl.i32],
    ):

        one_reg = tkw.Register[M, tkl.i32](1)
        res = tkw.atomic_add(one_reg, a, mapping=simple_read_mapping)
        tkw.write(res, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            B: 10,
            BLOCK_M: 64,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
    )
    wave_kernel = wave_compile(options, wave_kernel)
    if is_debug:
        print(wave_kernel.asm)

    # generate random input tensors between -1 and 1
    a = torch.randint(1, 2, (64,), dtype=torch.int32).cuda()
    c = torch.zeros((64,), dtype=torch.int32).cuda()

    wave_kernel(a, c)
    print(a)
    print(c)


def test_read_back_scalar(is_debug=False):
    """Perform atomic add to shared memory then read back a scalar value using dynamic mapping."""
    ONE = tkl.sym.ONE
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 0, M: 64, ONE: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(ONE, ONE, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(ONE, ONE)]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    simple_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: i},
        outputs={M: i},
    )

    dynamic_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: d0},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    @tkw.wave(constraints)
    def wave_kernel(
        a: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[ONE, ADDRESS_SPACE, tkl.i32],
    ):

        tid = tkw.scalar(THREAD_0, tkl.i32)
        one_reg = tkw.Register[M, tkl.i32](1)
        res = tkw.atomic_add(
            one_reg, a, mapping=dynamic_read_mapping, mapping_dynamic_vals=(tid,)
        )
        val = tkw.read(
            res,
            mapping=dynamic_read_mapping,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )
        tkw.write(val, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            ONE: 1,
            BLOCK_M: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
        minimize_shared_allocs=False,
    )
    wave_kernel = wave_compile(options, wave_kernel)
    if is_debug:
        print(wave_kernel.asm)

    # generate random input tensors between -1 and 1
    a = torch.randint(1, 2, (64,), dtype=torch.int32).cuda()
    b = torch.zeros((64,), dtype=torch.int32).cuda()
    c = torch.zeros((1,), dtype=torch.int32).cuda()

    wave_kernel(a, b, c)
    print(a)
    print(b)
    print(c)


def test_histogram(is_debug=False):
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS

    """Atomic add operation to a histogram using dynamic mapping."""
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: M, NUM_EXPERTS: NUM_EXPERTS},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, M, 0)]
    constraints += [tkw.WorkgroupConstraint(NUM_EXPERTS, NUM_EXPERTS, 1)]
    constraints += [tkw.WaveConstraint(M, M)]
    constraints += [tkw.WaveConstraint(NUM_EXPERTS, NUM_EXPERTS)]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    topk_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: d0},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    expert_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    @tkw.wave(constraints)
    def histogram_atomic_add(
        topk_ids: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        experts: tkl.Memory[NUM_EXPERTS, ADDRESS_SPACE, tkl.i32],
    ):
        one_reg = tkw.Register[NUM_EXPERTS, tkl.i32](1)
        tid = tkw.scalar(THREAD_0, tkl.i32)

        zero_vec = tkl.Register[NUM_EXPERTS, tkl.i32](0)
        shmem = tkw.allocate(
            shape=(NUM_EXPERTS,),
            distributed_shape=(NUM_EXPERTS,),
            dtype=tkl.i32,
        )
        tkw.write(zero_vec, shmem)

        expert_id = tkw.read(
            topk_ids,
            mapping=topk_read_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        tkw.atomic_add(
            one_reg,
            shmem,
            mapping=expert_read_map,
            mapping_dynamic_vals=(expert_id,),
            elements_per_thread=1,
        )

        tmp = tkw.read(shmem)
        tkw.write(tmp, experts)

    num_experts = 10
    num_tokens = 64
    hyperparams = {
        M: num_tokens,
        NUM_EXPERTS: num_experts,
    }
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        minimize_shared_allocs=False,
        print_ir_after="all" if is_debug else [],
    )
    histogram_atomic_add = wave_compile(options, histogram_atomic_add)
    if is_debug:
        print(histogram_atomic_add.asm)

    topk_ids = torch.randint(0, num_experts, (num_tokens,), dtype=torch.int32).cuda()
    experts = torch.zeros((num_experts,), dtype=torch.int32).cuda()
    histogram_atomic_add(topk_ids, experts)
    print("topk_ids: ", topk_ids)
    print("experts: ", experts)
    print("expected experts: ", torch.bincount(topk_ids, minlength=num_experts))


def test_large_histogram(is_debug=False):
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS
    TOKEN_OFFSET = tkl.sym.TOKEN_OFFSET
    """Atomic add operation to a histogram using dynamic mapping."""
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, M, 0)]
    constraints += [tkw.WorkgroupConstraint(NUM_EXPERTS, NUM_EXPERTS, 1)]
    constraints += [tkw.WaveConstraint(M, M)]
    constraints += [tkw.WaveConstraint(NUM_EXPERTS, NUM_EXPERTS)]

    constraints += [tkw.TilingConstraint(TOKEN_OFFSET)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: M, NUM_EXPERTS: NUM_EXPERTS, TOKEN_OFFSET: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    topk_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={M: d0},
        outputs={M: i},
        dynamic_val_mappings={M: i},
    )

    expert_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    @tkw.wave(constraints)
    def histogram_atomic_add(
        topk_ids: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        experts: tkl.Memory[NUM_EXPERTS, ADDRESS_SPACE, tkl.i32],
    ):
        one_reg = tkw.Register[NUM_EXPERTS, tkl.i32](1)
        zero_reg = tkw.Register[TOKEN_OFFSET, tkl.i32](0)

        loop_condition = TOKEN_OFFSET < M

        @tkw.iterate(
            TOKEN_OFFSET, start=zero_reg, condition=loop_condition, init_args=[]
        )
        def count_tokens():
            token_idx = tkw.self_index(TOKEN_OFFSET, tkl.i32)
            tid_reg = tkw.Register[TOKEN_OFFSET, tkl.i32](THREAD_0)
            token_idx = token_idx * tkl.Register[TOKEN_OFFSET, tkl.i32](64) + tid_reg

            expert_id = tkw.read(
                topk_ids,
                mapping=topk_read_map,
                mapping_dynamic_vals=(token_idx,),
                elements_per_thread=1,
            )

            tkw.atomic_add(
                one_reg,
                experts,
                mapping=expert_read_map,
                mapping_dynamic_vals=(expert_id,),
                elements_per_thread=1,
            )

            next_token_idx = token_idx + tkl.Register[TOKEN_OFFSET, tkl.i32](64)
            tkw.set_symbol(TOKEN_OFFSET, next_token_idx)

    num_experts = 10
    num_tokens = 64
    hyperparams = {
        M: num_tokens,
        NUM_EXPERTS: num_experts,
    }
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        minimize_shared_allocs=False,
        print_ir_after="all" if is_debug else [],
    )
    histogram_atomic_add = wave_compile(options, histogram_atomic_add)
    if is_debug:
        print(histogram_atomic_add.asm)

    topk_ids = torch.randint(0, num_experts, (num_tokens,), dtype=torch.int32).cuda()
    experts = torch.zeros((num_experts,), dtype=torch.int32).cuda()

    histogram_atomic_add(topk_ids, experts)
    print("topk_ids: ", topk_ids)
    print("experts: ", experts)
    print("expected experts: ", torch.bincount(topk_ids, minlength=num_experts))


if __name__ == "__main__":
    args = parse_args()
    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(args.test, globals(), args.debug, args.repeat)
    exit(0 if success else 1)
