Index-Based Multibuffering
==========================

Overview
--------

**Multibuffering** is a software pipelining optimization that increases
instruction-level parallelism by having different iterations use different
sections of shared memory so that pipelined kernels do not have hazards.

Index based multibuffering involves changing shared-memory buffers to increase
their allocated size, then modify indices into these buffers so that each iteration
accesses a different subset of these buffers.

For example if iterations i and i+1 both read and write to a shared buffer B of
size B for intermediate steps during a kernel, during a pipelined schedule both
iterations would have a hazard with each other when instructions from iteration i
and i+1 become overlapped.

An index based multibuffering optimization would instead increase the dimensions
of B to n*B, where n is the multibuffer count. It would also use the iteration
indices to determine that i would access the region from B[0:B], i+1 would access
the region from B[B:2B], etc for n different regions.

Crucially, each loop iteration operates on a *different* copy of the buffer, which removes
the need for LDS (local data share) barriers between unrelated iterations.

By eliminating these barriers, the scheduler can potentially freely reorder instructions
and overlap memory accesses with computation, unlocking additional
latency-hiding opportunities. See four_stage_scheduler.rst for an example multibuffered
implementation with two buffers.


Implementation Details
----------------------

This implementation identifies shared-memory reads/writes and **doubles** (or
otherwise scales) the relevant buffer dimension according to the
``multi_buffer_count``. The index expression is then rewritten as:

.. math::

    \text{new\_index} = \text{original\_index} + \text{block\_size} \times (\text{induction\_var} \bmod \text{multi\_buffer\_count})

Key steps:

1. **Detection** — Identify shared-memory reads/writes and gather-to-LDS
   operations.
2. **Dimension Scaling** — Increase the buffer dimension size by
   ``multi_buffer_count``.
3. **Index Rewrite** — Offset the index based on the induction variable modulo
   the buffer count.
4. **Mapping Update** — Update index mappings so that later schedule
   transformations (prologue/steady/epilogue phases) see consistent shapes.



Usage in Scheduler
------------------

Index-based multibuffering is automatically enabled when
``multi_buffer_count`` is set in `WaveCompileOptions`. For example:

.. code-block:: python

    options = WaveCompileOptions(
        schedule=SchedulingType.FOUR_STAGE,
        multi_buffer_count=2,
    )

This optimization is especially effective in conjunction with
`FourStageScheduler` and modulo scheduling, where overlapping memory
access and compute is critical to achieving initiation interval = 1.
