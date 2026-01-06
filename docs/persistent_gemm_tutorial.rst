Persistent GEMM Tutorial
========================
This tutorial builds upon GEMM tutorial to implement a **persistent GEMM kernel**. We'll explain what persistence means, why it's beneficial, and walk through the implementation step by step.

What is a Persistent Kernel?
----------------------------

In a traditional GEMM kernel, there's a 1:1 mapping between workgroups and output tiles:

- Workgroup 0 computes tile (0, 0)
- Workgroup 1 computes tile (0, 1)
- And so on...

The GPU scheduler launches exactly as many workgroups as there are output tiles. For large matrices, this can mean launching thousands of workgroups.

A **persistent kernel** takes a different approach:

- Launch a fixed number of workgroups (typically matching the number of compute units)
- Each workgroup processes **multiple tiles** in a loop
- Workgroups "persist" on the GPU, iterating until all tiles are complete

Benefits of Persistence
~~~~~~~~~~~~~~~~~~~~~~~

1. **Reduced Launch Overhead**: Fewer workgroups to schedule
2. **Improve Occupancy Control**: Match workgroups to hardware compute units with less hardware divergence

Implementation
--------------

Let's build a persistent GEMM kernel step by step.

Step 1: Imports and Symbols
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond the standard GEMM symbols, let's define additional symbols for persistence:

.. code-block:: python

    from wave_lang.kernel._support.indexing import sym
    from wave_lang.kernel._support.dtype import f16, f32, i32
    from wave_lang.kernel.lang.wave_types import *
    from wave_lang.kernel.lang.global_symbols import *
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
    import torch
    from sympy import ceiling, floor

    # Standard GEMM symbols
    M = sym.M
    N = sym.N
    K = sym.K
    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N
    BLOCK_K = sym.BLOCK_K
    ADDRESS_SPACE = sym.ADDRESS_SPACE

    # Persistent kernel symbols
    TOTAL_TILES = sym.TOTAL_TILES      # Total number of output tiles
    NUM_CTAS = sym.NUM_CTAS            # Number of persistent workgroups
    TILE_IDX = sym.TILE_IDX            # Current tile index (loop variable)
    CTA_M_OFFSET = sym.CTA_M_OFFSET    # M offset for current tile
    CTA_N_OFFSET = sym.CTA_N_OFFSET    # N offset for current tile
    N_TILES = sym.N_TILES              # Number of tiles in N dimension

Step 2: Index Mappings
~~~~~~~~~~~~~~~~~~~~~~

Since each workgroup processes multiple tiles, we need to manually control memory access offsets using ``IndexMapping``:

.. code-block:: python

    # Create iterator symbols for mappings
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    # Reading A: add CTA_M_OFFSET to the M dimension
    a_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i + CTA_M_OFFSET, K: j},
        outputs={M: i, K: j},
    )

    # Reading B: add CTA_N_OFFSET to the N dimension
    b_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={N: i + CTA_N_OFFSET, K: j},
        outputs={N: i, K: j},
    )

    # Writing C: add both offsets
    c_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
    )

The ``inputs`` specify the source coordinates in memory, while ``outputs`` specify the logical coordinates of the section in memory we want to return.
The offset symbols (``CTA_M_OFFSET``, ``CTA_N_OFFSET``) are set dynamically during execution.

Step 3: Constraints with GridConstraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key difference from vanilla GEMM is the ``GridConstraint``:

.. code-block:: python

    constraints = [
        # GridConstraint specifies the number of persistent workgroups
        tkw.GridConstraint(NUM_CTAS),

        # Standard workgroup constraints (still needed for shared memory allocation)
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),

        # K dimension tiling for the inner reduction loop
        tkw.TilingConstraint(K, BLOCK_K),

        # TILE_IDX tiling for the outer persistent loop
        tkw.TilingConstraint(TILE_IDX),

        # Wave-level parallelism
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),

        # Hardware constraints
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={TILE_IDX: 0},  # TILE_IDX is a scalar, not vectorized
        ),
    ]


Step 4: The Persistent Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @tkw.wave(constraints)
    def persistent_gemm(
        a: Memory[M, K, ADDRESS_SPACE, f16],
        b: Memory[N, K, ADDRESS_SPACE, f16],
        c: Memory[M, N, GLOBAL_ADDRESS_SPACE, f32],
    ):
        # Loop condition: continue while tile index < total tiles
        condition = TILE_IDX < TOTAL_TILES

        # Each workgroup starts at its workgroup ID
        init_tile_id = tkw.scalar(WORKGROUP_0, tkl.i32)

        # Outer persistent loop - iterates over tiles
        @tkw.iterate(TILE_IDX, start=init_tile_id, condition=condition, init_args=[])
        def persistent_loop():
            # Get current tile index
            tile_id = tkw.self_index(TILE_IDX, tkl.i32)

            # Convert linear tile_id to 2D tile coordinates (row-major order)
            m_offset = (tile_id // tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_M, i32)
            n_offset = (tile_id % tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_N, i32)

            # Set the offset symbols for index mappings
            tkw.set_symbol(CTA_M_OFFSET, m_offset)
            tkw.set_symbol(CTA_N_OFFSET, n_offset)

            # Initialize accumulator for this tile
            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            # Inner K-loop (same as vanilla GEMM)
            @tkw.iterate(axis=K, init_args=[c_reg])
            def k_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, mapping=a_read_mapping)
                b_reg = tkw.read(b, mapping=b_read_mapping)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            # Write result for this tile
            tkw.write(k_loop, c, mapping=c_write_mapping)

            # Advance to next tile: tile_id += NUM_CTAS
            num_cus_scalar = tkw.scalar(NUM_CTAS, tkl.i32)
            next_idx = tile_id + num_cus_scalar
            tkw.set_symbol(TILE_IDX, next_idx)

Understanding the Tile Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tile distribution follows a strided pattern:

- Workgroup 0 processes tiles: 0, NUM_CTAS, 2*NUM_CTAS, ...
- Workgroup 1 processes tiles: 1, NUM_CTAS+1, 2*NUM_CTAS+1, ...
- And so on...

For example, with 4 workgroups and 10 tiles::

    Workgroup 0: tiles 0, 4, 8
    Workgroup 1: tiles 1, 5, 9
    Workgroup 2: tiles 2, 6
    Workgroup 3: tiles 3, 7


L2 Cache Swizzling
----------------------------

Row-major tile ordering isn't optimal for L2 cache utilization. Adjacent tiles in the linear order may access distant memory regions.

**L2 Swizzling** groups nearby tiles together to improve data reuse.

.. code-block:: python

    # Additional symbols for swizzling
    M_TILES = sym.M_TILES
    GROUP_SIZE_M = sym.GROUP_SIZE_M
    NUM_CTAS_IN_GROUP = sym.NUM_CTAS_IN_GROUP

    # Group tiles for L2 locality
    group_id = tile_id // tkw.scalar(NUM_CTAS_IN_GROUP, i32)
    first_cta_m = group_id * tkw.scalar(GROUP_SIZE_M, i32)
    group_size_m = tkw.minimum(
        tkw.scalar(M_TILES, i32) - first_cta_m,
        tkw.scalar(GROUP_SIZE_M, i32)
    )

    # Compute swizzled coordinates
    cta_m = first_cta_m + ((tile_id % tkw.scalar(NUM_CTAS_IN_GROUP, i32)) % group_size_m)
    cta_n = (tile_id % tkw.scalar(NUM_CTAS_IN_GROUP, i32)) // group_size_m

    m_offset = cta_m * tkw.scalar(BLOCK_M, i32)
    n_offset = cta_n * tkw.scalar(BLOCK_N, i32)

This groups tiles into rectangular regions (GROUP_SIZE_M x N_TILES), improving spatial locality in L2 cache.

XCD Swizzling (LLC/MALL Cache)
------------------------------

The AMD MI300 series features a chiplet architecture with multiple XCDs (Accelerated Compute Dies), and the Last Level Cache (LLC), also known as the MALL (Memory Attached Last Level) cache.

By default, the hardware assigns workgroups to XCDs in a round-robin fashion:

- Workgroup 0 → XCD 0
- Workgroup 1 → XCD 1
- ...
- Workgroup 7 → XCD 7
- Workgroup 8 → XCD 0 (wraps around)

This default scheduling scatters related workgroups across different XCDs, preventing them from sharing data in the LLC.

**XCD Swizzling** remaps workgroup IDs so that workgroups processing nearby tiles are scheduled on the same XCD, improving LLC cache reuse.

.. code-block:: python

    # Additional symbols for XCD swizzling
    NUM_XCDS = sym.NUM_XCDS      # Number of XCDs (8 for MI350)
    CHUNK_SIZE = sym.CHUNK_SIZE  # Number of consecutive tiles per XCD

    @tkw.wave(constraints)
    def persistent_gemm(
        a: Memory[M, K, ADDRESS_SPACE, f16],
        b: Memory[N, K, ADDRESS_SPACE, f16],
        c: Memory[M, N, GLOBAL_ADDRESS_SPACE, f32],
    ):
        cta = tkw.scalar(WORKGROUP_0, i32)

        # XCD Swizzling: remap workgroup ID for better LLC locality
        xcd_id = cta % tkw.scalar(NUM_XCDS, i32)
        local_cta = cta // tkw.scalar(NUM_XCDS, i32)
        chunk_idx = local_cta // tkw.scalar(CHUNK_SIZE, i32)
        pos_in_chunk = local_cta % tkw.scalar(CHUNK_SIZE, i32)

        # Reorder so consecutive chunks go to the same XCD
        remapped_cta = (chunk_idx * tkw.scalar(NUM_XCDS, i32) * tkw.scalar(CHUNK_SIZE, i32)
                       + xcd_id * tkw.scalar(CHUNK_SIZE, i32)
                       + pos_in_chunk)

        # Handle tiles that don't fit evenly into chunks
        chunked_threshold = ((tkw.scalar(TOTAL_TILES, i32)
                             // (tkw.scalar(NUM_XCDS, i32) * tkw.scalar(CHUNK_SIZE, i32)))
                            * (tkw.scalar(NUM_XCDS, i32) * tkw.scalar(CHUNK_SIZE, i32)))
        in_chunked_region = cta <= chunked_threshold
        init_tile_id = tkw.select(in_chunked_region, remapped_cta, cta)

        @tkw.iterate(TILE_IDX, start=init_tile_id, condition=condition, init_args=[])
        def persistent_loop():
            # ... (L2 swizzling and GEMM computation)

How XCD Swizzling Works
~~~~~~~~~~~~~~~~~~~~~~~

Without swizzling (round-robin), workgroups are scattered::

    XCD 0: workgroups 0, 8, 16, 24, ...
    XCD 1: workgroups 1, 9, 17, 25, ...
    ...

With swizzling (CHUNK_SIZE=2), consecutive workgroups are grouped on the same XCD::

    XCD 0: workgroups 0, 1, 16, 17, ...  (chunks 0, 2, ...)
    XCD 1: workgroups 2, 3, 18, 19, ...  (chunks 1, 3, ...)
    ...

This ensures that workgroups processing adjacent tiles share the same LLC, improving data reuse for both input matrices A and B.

Combining L2 and XCD Swizzling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance on the MI300 series, combine both optimizations:

1. **XCD Swizzling**: Remaps ``init_tile_id`` so nearby tiles run on the same XCD
2. **L2 Swizzling**: Converts ``tile_id`` to 2D coordinates that maximize L2 locality

This approach optimizes for both the LLC and L2 cache.

Full Kernel with Both Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is the complete persistent GEMM kernel combining both L2 and XCD swizzling:

.. code-block:: python

    from wave_lang.kernel._support.indexing import sym
    from wave_lang.kernel._support.dtype import f16, f32, i32
    from wave_lang.kernel.lang.wave_types import *
    from wave_lang.kernel.lang.global_symbols import *
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
    import torch

    # Standard GEMM symbols
    M = sym.M
    N = sym.N
    K = sym.K
    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N
    BLOCK_K = sym.BLOCK_K
    ADDRESS_SPACE = sym.ADDRESS_SPACE

    # Persistent kernel symbols
    TOTAL_TILES = sym.TOTAL_TILES
    NUM_CTAS = sym.NUM_CTAS
    TILE_IDX = sym.TILE_IDX
    CTA_M_OFFSET = sym.CTA_M_OFFSET
    CTA_N_OFFSET = sym.CTA_N_OFFSET

    # L2 swizzling symbols
    M_TILES = sym.M_TILES
    N_TILES = sym.N_TILES
    GROUP_SIZE_M = sym.GROUP_SIZE_M
    NUM_CTAS_IN_GROUP = sym.NUM_CTAS_IN_GROUP

    # XCD swizzling symbols
    NUM_XCDS = sym.NUM_XCDS
    CHUNK_SIZE = sym.CHUNK_SIZE

    # Index mappings for manual offset control
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    a_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i + CTA_M_OFFSET, K: j},
        outputs={M: i, K: j},
    )

    b_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={N: i + CTA_N_OFFSET, K: j},
        outputs={N: i, K: j},
    )

    c_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
    )

    constraints = [
        tkw.GridConstraint(NUM_CTAS),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.TilingConstraint(TILE_IDX),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={TILE_IDX: 0},
        ),
    ]


    @tkw.wave(constraints)
    def persistent_gemm(
        a: Memory[M, K, ADDRESS_SPACE, f16],
        b: Memory[N, K, ADDRESS_SPACE, f16],
        c: Memory[M, N, GLOBAL_ADDRESS_SPACE, f32],
    ):
        cta = tkw.scalar(WORKGROUP_0, i32)

        # XCD Swizzling (LLC/MALL Cache)
        xcd_id = cta % tkw.scalar(NUM_XCDS, i32)
        local_cta = cta // tkw.scalar(NUM_XCDS, i32)
        chunk_idx = local_cta // tkw.scalar(CHUNK_SIZE, i32)
        pos_in_chunk = local_cta % tkw.scalar(CHUNK_SIZE, i32)
        remapped_cta = (chunk_idx * tkw.scalar(NUM_XCDS, i32) * tkw.scalar(CHUNK_SIZE, i32)
                       + xcd_id * tkw.scalar(CHUNK_SIZE, i32) + pos_in_chunk)

        # Select: use remapped_cta if in chunked region, else use original cta
        chunked_threshold = ((tkw.scalar(TOTAL_TILES, i32)
                             // (tkw.scalar(NUM_XCDS, i32) * tkw.scalar(CHUNK_SIZE, i32)))
                            * (tkw.scalar(NUM_XCDS, i32) * tkw.scalar(CHUNK_SIZE, i32)))
        in_chunked_region = cta <= chunked_threshold
        init_tile_id = tkw.select(in_chunked_region, remapped_cta, cta)

        condition = TILE_IDX < TOTAL_TILES

        @tkw.iterate(TILE_IDX, start=init_tile_id, condition=condition, init_args=[])
        def persistent_loop():
            tile_id = tkw.self_index(TILE_IDX, tkl.i32)

            # L2 Cache Swizzling
            group_id = tile_id // tkw.scalar(NUM_CTAS_IN_GROUP, i32)
            first_cta_m = group_id * tkw.scalar(GROUP_SIZE_M, i32)
            group_size_m = tkw.minimum(
                tkw.scalar(M_TILES, i32) - first_cta_m,
                tkw.scalar(GROUP_SIZE_M, i32)
            )
            cta_m = first_cta_m + ((tile_id % tkw.scalar(NUM_CTAS_IN_GROUP, i32)) % group_size_m)
            cta_n = (tile_id % tkw.scalar(NUM_CTAS_IN_GROUP, i32)) // group_size_m

            m_offset = cta_m * tkw.scalar(BLOCK_M, i32)
            n_offset = cta_n * tkw.scalar(BLOCK_N, i32)
            tkw.set_symbol(CTA_M_OFFSET, m_offset)
            tkw.set_symbol(CTA_N_OFFSET, n_offset)

            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(axis=K, init_args=[c_reg])
            def k_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, mapping=a_read_mapping)
                b_reg = tkw.read(b, mapping=b_read_mapping)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(k_loop, c, mapping=c_write_mapping)

            num_cus_scalar = tkw.scalar(NUM_CTAS, tkl.i32)
            next_idx = tile_id + num_cus_scalar
            tkw.set_symbol(TILE_IDX, next_idx)

Testing the Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a test function to verify our persistent GEMM implementation:

.. code-block:: python

    def test_persistent_gemm():
        # Test matrix dimensions
        m, n, k = 2048, 2048, 2048

        # Block sizes
        block_m, block_n, block_k = 128, 256, 64

        # Calculate tile grid
        m_tiles = (m + block_m - 1) // block_m
        n_tiles = (n + block_n - 1) // block_n
        total_tiles = m_tiles * n_tiles

        # Swizzling parameters
        group_size_m = 4
        num_xcds = 8
        num_ctas_in_group = group_size_m * n_tiles
        num_ctas = total_tiles
        chunk_size = 2

        print(f"Tile grid: {m_tiles} x {n_tiles} = {total_tiles} tiles")

        # Initialize input matrices with random values
        torch.manual_seed(0)
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(n, k, dtype=torch.float16, device="cuda")
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

        # Set hyperparameters for compilation
        hyperparams = {
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            BLOCK_M: block_m,
            BLOCK_N: block_n,
            BLOCK_K: block_k,
            M: m,
            N: n,
            K: k,
            TOTAL_TILES: total_tiles,
            M_TILES: m_tiles,
            N_TILES: n_tiles,
            GROUP_SIZE_M: group_size_m,
            NUM_CTAS_IN_GROUP: num_ctas_in_group,
            NUM_CTAS: num_ctas,
            NUM_XCDS: num_xcds,
            CHUNK_SIZE: chunk_size,
        }

        # Compile the kernel
        options = WaveCompileOptions(subs=hyperparams)
        options = set_default_run_config(options)
        compiled_gemm = wave_compile(options, persistent_gemm)

        # Run the GEMM kernel
        compiled_gemm(a, b, c)

        # Verify the result using PyTorch's matmul
        expected = torch.matmul(a, b.t())

        # Check if results are close (accounting for floating-point precision)
        assert torch.allclose(c.to(torch.float16), expected, rtol=1e-2, atol=1e-2), \
            f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

        print("Persistent GEMM test passed!")

Key Takeaways
-------------

1. **GridConstraint** controls the number of persistent workgroups
2. **IndexMapping** enables manual offset control for dynamic tile assignment
3. **set_symbol** updates offset symbols during the persistent loop
4. **self_index** retrieves the current iteration value
5. The outer loop sets a **start** and **condition** to setup the tile distribution for a single CTA
