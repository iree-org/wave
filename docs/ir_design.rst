Vector Shapes and Hardware Constraints
======================================

This document describes the ``vector_shapes`` field on
``#wave.hardware_constraint`` and how it relates to ``mma_type``,
``elements_per_thread``, and the constraint system in the Water IR.


Overview
--------

``vector_shapes`` is an optional ``DictionaryAttr`` on
``#wave.hardware_constraint``.  Each entry maps a dimension name (a string
matching a ``#wave.symbol``) to an integer specifying how many elements a
single wave processes along that dimension in one instance of an operation
before expansion has replicated it.

.. code-block:: mlir

   #wave.hardware_constraint<
       threads_per_wave = 64,
       waves_per_block = [2, 2, 1],
       mma_type = #wave.mma_kind<f32_16x16x16_f16>,
       vector_shapes = {M = 16, N = 16, K = 16},
       max_bits_per_load = 128>

``vector_shapes`` is the central piece of information the compiler uses to:

* distribute work across threads within a wave,
* determine how many elements each thread processes (``elements_per_thread``),
* compute memory access strides, and
* drive the expansion (unrolling) pass that replicates operations until the
  workgroup tile is covered.


Where vector_shapes comes from
-------------------------------

There are two cases, depending on whether ``mma_type`` is present.

**When mma_type is set,** ``vector_shapes`` is derived from the MMA
instruction geometry.  ``WaveMmaKindAttr::getShape`` returns the ``(M, N, K)``
tile for the intrinsic and those sizes become the vector shape entries:

.. code-block:: text

   mma_type = f32_16x16x16_f16  →  getShape = (16, 16, 16)
                                    vector_shapes = {M = 16, N = 16, K = 16}

Additional entries may be provided for dimensions the MMA analysis does not
cover (e.g. a batch dimension), and in that case both ``mma_type`` and explicit
``vector_shapes`` coexist.

**When mma_type is absent,** ``vector_shapes`` is specified directly or derived
from workgroup / tiling constraint tile sizes.  In either case it must be
present for the compiler to proceed.

In MLIR, ``vector_shapes`` entries must all be ``IntegerAttr`` values.  The
verifier in ``WaveDialect.cpp`` enforces this.


The special value 0
^^^^^^^^^^^^^^^^^^^

A vector shape of ``0`` marks a dimension as *scalar* — the wave does not tile
along it.  This is used for dimensions like batch (``B``) that should not
contribute to the intra-wave data distribution:

.. code-block:: mlir

   vector_shapes = {B = 0, M = 16, N = 16}


Relationship to workgroup and tiling constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``vector_shapes`` and constraint tile sizes serve different purposes:

* **Tile size** (from ``#wave.workgroup_constraint`` or
  ``#wave.tiling_constraint``) is the total amount of work assigned to one
  workgroup or one iteration of a reduction loop along a dimension.
* **Vector shape** is the amount of work one wave handles in a single instance
  of an operation along that dimension.

**When mma_type is present,** the vector shapes derive from the MMA geometry
and are typically smaller than the constraint tile sizes.  The expansion pass
(which runs on the Python/FX side) replicates each
operation to cover the tile.  For example, with ``BLOCK_M = 64``,
``waves_per_block = [2, 2, 1]``, and ``mma_type = f32_16x16x16_f16``
(vector shape 16 for M):

.. code-block:: text

   expansion_count = ceil(64 / (2 × 16)) = 2

The MLIR IR only sees the already-expanded result: two ``wave.mma`` ops along M
rather than one.  The ``vector_shapes`` remain on the
``#wave.hardware_constraint`` for verification and for passes that need to
reason about the per-wave tile.

**When mma_type is absent,** the MLIR verifier enforces that each
``vector_shapes`` entry **matches** the resolved tile size from the
corresponding ``#wave.workgroup_constraint`` or ``#wave.tiling_constraint`` for
that dimension. Unlike with mma_operations that have a fixed size, element wise operations
can operate on any number of elements_per_thread and thus don't need to be expanded multiple times.
A mismatch is a verification error:

.. code-block:: mlir

   // ERROR: vector_shapes entry 'M' (16) does not match
   //        workgroup constraint tile size (32)
   #wave.hardware_constraint<threads_per_wave = 64, vector_shapes = {M = 16}>

This means that in non-MMA programs, there is no separate expansion step:
``vector_shapes`` equals the tile size and each operation appears exactly once
per dimension.


MMA kind and intrinsic shapes
------------------------------

``WaveMmaKindEnum`` enumerates hardware matrix multiply intrinsics.  Each
variant encodes the output element type, tile shape (M×N×K), and input element
type.  Examples:

.. code-block:: mlir

   #wave.mma_kind<f32_16x16x16_f16>               // (M=16, N=16, K=16)
   #wave.mma_kind<f32_32x32x8_f16>                 // (M=32, N=32, K=8)
   #wave.mma_kind<f32_16x16x128_f8f6f4>            // (M=16, N=16, K=128)

``WaveMmaKindAttr::getShape(ctx, kind)`` returns the ``(M, N, K)`` tuple.

The ``kind`` attribute on ``wave.mma`` may differ from the ``mma_type`` on the
hardware constraint.  When ``kind`` is absent, the
``PropagateDefaultsFromConstraints`` pass fills it from the hardware
constraint's ``mma_type``.  When multiple ``wave.mma`` ops exist in the same
function, each carries its own ``kind`` and its own effective vector shapes.


Relationship to elements_per_thread
-------------------------------------

``elements_per_thread`` is an optional ``I64Attr`` on ``wave.read`` and
``wave.write``.  It specifies how many contiguous elements a single thread
loads or stores in one operation instance:

.. code-block:: mlir

   %0 = wave.read %mem { elements_per_thread = 8 }
       : (!wave.tensor<[@M, @K] of f16, <global>>)
       -> !wave.tensor<[@M, @K] of f16, <register>>

``elements_per_thread`` is related to ``vector_shapes`` conceptually: the
vector shape for a dimension gives the total elements a wave handles, and
dividing by ``threads_per_wave`` (for a reduction dimension) or accounting for
thread count per workgroup dimension gives the per-thread count.  The
``PropagateElementsPerThread`` pass can infer ``elements_per_thread`` from the
hardware constraint when it is not explicitly provided.
