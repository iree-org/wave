Wave Schedule Construct
======================

The ``wave_schedule`` construct is a powerful new feature in the Wave language that enables kernel authors to explicitly manipulate the execution graph and perform advanced optimizations like pipelining. This construct provides a declarative way to define custom scheduling strategies while maintaining the ability to verify schedules before execution through tracing.

Overview
--------

The ``wave_schedule`` decorator allows developers to define custom scheduling logic that operates on the kernel's execution graph. Unlike traditional scheduling approaches that rely on automatic heuristics, ``wave_schedule`` gives kernel authors fine-grained control over how operations are scheduled and executed.

Key Features
~~~~~~~~~~~~

- **Explicit Graph Manipulation**: Direct control over operation scheduling and execution order
- **Tracing-Based Verification**: User-provided schedules can be verified before execution
- **Tag-Based Reference System**: Operators can be referenced in schedules using tags

How It Works
------------

The ``wave_schedule`` construct works through a combination of tracing and custom schedule operations:

1. **Tracing Phase**: The schedule function is traced to capture the intended scheduling operations
2. **Verification**: The traced schedule can be analyzed and verified before execution
3. **Application**: The verified schedule is applied to the kernel's execution graph
4. **Execution**: The kernel runs with the custom schedule applied

Schedule Operations
-------------------

The ``wave_schedule`` construct provides several built-in operations for manipulating the execution graph:

get_node_by_tag
~~~~~~~~~~~~~~~

Retrieves nodes from the kernel trace by their tag:

.. code-block:: python

    k_loop = wave.get_node_by_tag("k_loop")

This operation finds all nodes in the kernel that have been tagged with the specified string.

get_node_by_tag_and_type
~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieves nodes by both tag and type, providing more precise selection:

.. code-block:: python

    load_a = wave.get_node_by_tag_and_type("read_a", wave.Read)

This is useful when you need to select specific types of operations (like reads or writes) from a tagged group.

partition_by_address_space
~~~~~~~~~~~~~~~~~~~~~~~~~~

Partitions nodes based on their memory address space:

.. code-block:: python

    global_load_a, shared_load_a = wave.partition_by_address_space(
        load_a, GLOBAL_ADDRESS_SPACE
    )

This operation separates operations based on whether they access global or shared memory, enabling different scheduling strategies for different memory types.

pipeline
~~~~~~~~

Creates a pipelined execution pattern with automatic stage and initiation interval calculation:

.. code-block:: python

    with wave.pipeline(k_loop) as pipelined_loop:
        pipelined_loop.set_stage([
            (global_load_a, global_load_b),
            (shared_write_a, shared_write_b),
        ])
        pipelined_loop.set_stage([
            (shared_load_a, shared_load_b),
            (mma,),
        ])

The pipeline operation enables advanced optimizations like overlapping memory operations with computation.

**Automatic Parameter Calculation**: The pipeline system automatically calculates the initiation interval based on the number of operation groups (tuples) in each stage. For example, if a stage contains 2 operation groups like ``[(op1, op2), (op3, op4)]``, the initiation interval for that stage will be 2. Stages are numbered sequentially (0, 1, 2, ...) as they are added to the pipeline.

**Operation Grouping**: Operations within each tuple are scheduled to execute in parallel, while tuples within a stage are scheduled sequentially. Additionally, operations in the same tuple position (ith tuple) across different stages will also co-execute. This allows fine-grained control over which operations can overlap and which must execute in order.

For example, in a 2-stage pipeline where stage 0 has ``[(op1, op2), (op3, op4)]`` and stage 1 has ``[(op5, op6), (op7, op8)]``:
- ``op1`` and ``op2`` execute in parallel (same tuple in stage 0)
- ``op3`` and ``op4`` execute in parallel (same tuple in stage 0)
- ``op5`` and ``op6`` execute in parallel (same tuple in stage 1)
- ``op7`` and ``op8`` execute in parallel (same tuple in stage 1)
- ``op1``, ``op2``, ``op5``, and ``op6`` co-execute in parallel (all are the 0th tuple across stages)
- ``op3``, ``op4``, ``op7``, and ``op8`` co-execute in parallel (all are the 1st tuple across stages)

getitem
~~~~~~~

Provides array-like access to collections of nodes:

.. code-block:: python

    first_node = wave.getitem(node_collection, 0)

This is useful for accessing specific elements from operations that return collections.

Tag System
----------

The tag system is fundamental to how ``wave_schedule`` works. Tags allow operators in the original kernel to be referenced and manipulated in the schedule:

Adding Tags to Kernel Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tags are added to kernel operations using the ``tag`` parameter:

.. code-block:: python

    @wave.iterate(K, init_args=[c_reg], tag="k_loop")
    def repeat(acc):
        return acc

    a_reg = wave.read(a, tag="read_a")
    b_reg = wave.read(b, tag="read_b")
    c_reg = wave.mma(a_reg, b_reg, tag="mma")

Referencing Tagged Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once tagged, operations can be referenced in the schedule:

.. code-block:: python

    @wave_schedule.wave_schedule()
    def custom_schedule():
        k_loop = wave.get_node_by_tag("k_loop")
        load_a = wave.get_node_by_tag_and_type("read_a", wave.Read)
        mma = wave.get_node_by_tag("mma")

Minimal Kernel Modifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key advantage of the tag system is that it requires minimal modifications to the original kernel. The only changes needed are:

1. Adding ``tag`` parameters to operations that need to be referenced
2. The rest of the kernel logic remains unchanged

This makes it easy to add custom scheduling to existing kernels without major refactoring.


Example Usage
-------------

Here's a complete example showing how to use ``wave_schedule`` for a GEMM kernel:

Kernel Definition
~~~~~~~~~~~~~~~~~

.. code-block:: python

    @wave.wave(constraints)
    def gemm_prefetch(
        a: Memory[M, K, ADDRESS_SPACE_0, f16],
        b: Memory[K, N, ADDRESS_SPACE_0, f16],
        c: Memory[M, N, ADDRESS_SPACE, f16],
    ):
        c_reg = Register[M, N, f32](0.0)

        @wave.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc):
            a_reg = wave.read(a, tag="read_a")
            b_reg = wave.read(b, tag="read_b")
            repeat = wave.mma(a_reg, b_reg, acc, tag="mma")
            return repeat

        wave.write(repeat, c)

Schedule Definition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @wave_schedule.wave_schedule()
    def prefetch_schedule():
        # Get nodes to be manipulated in the schedule
        k_loop = wave.get_node_by_tag("k_loop")
        load_a = wave.get_node_by_tag_and_type("read_a", wave.Read)
        global_load_a, shared_load_a = wave.partition_by_address_space(
            load_a, GLOBAL_ADDRESS_SPACE
        )
        shared_write_a = wave.get_node_by_tag_and_type("read_a", wave.Write)
        load_b = wave.get_node_by_tag_and_type("read_b", wave.Read)
        global_load_b, shared_load_b = wave.partition_by_address_space(
            load_b, GLOBAL_ADDRESS_SPACE
        )
        shared_write_b = wave.get_node_by_tag_and_type("read_b", wave.Write)
        mma = wave.get_node_by_tag("mma")

        # Create a pipeline with 2 stages and automatic initiation interval calculation
        with wave.pipeline(k_loop) as pipelined_loop:
            pipelined_loop.set_stage([
                (global_load_a, global_load_b),
                (shared_write_a, shared_write_b),
            ])
            pipelined_loop.set_stage([
                (shared_load_a, shared_load_b),
                (mma,),
            ])

**Co-execution in GEMM Example**: In this pipeline configuration, the following clusters will co-execute:

- **Stage 0, Tuple 0**: Global memory loads for matrices A and B execute in parallel
- **Stage 0, Tuple 1**: Shared memory writes for matrices A and B execute in parallel
- **Stage 1, Tuple 0**: Shared memory loads for matrices A and B execute in parallel
- **Stage 1, Tuple 1**: MMA computation executes alone

**Cross-stage co-execution**:
- Global memory load cluster (stage 0, tuple 0) co-executes with shared memory load cluster (stage 1, tuple 0)
- Shared memory write cluster (stage 0, tuple 1) co-executes with MMA computation cluster (stage 1, tuple 1)

This creates an efficient pipeline where global memory loads overlap with shared memory loads, and shared memory writes overlap with MMA computation.

Compilation and Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    options = WaveCompileOptions(
        subs={...},
        schedule=SchedulingType.MANUAL,
        use_scheduling_barriers=True,
        compile_to_mlir=True,
    )

    compiled_kernel = wave_compile(options, gemm_prefetch, prefetch_schedule)

This example demonstrates:

1. **Tagging**: Operations are tagged with meaningful names
2. **Node Selection**: Schedule operations select and partition nodes
3. **Pipeline Creation**: A 2-stage pipeline is created with automatic initiation interval calculation based on the number of operation groups in each stage
4. **Operation Grouping**: Operations are grouped using tuples to specify which operations can execute in parallel within each stage
5. **Manual Scheduling**: The schedule is applied using ``SchedulingType.MANUAL`` (and in this case produces the same result as setting the schedule to ``SchedulingType.PREFETCH``)


Conclusion
----------

The ``wave_schedule`` construct provides a powerful and flexible way to implement custom scheduling strategies in Wave kernels. By combining explicit control with verification capabilities, it enables kernel authors to achieve optimal performance while maintaining correctness guarantees.
