========================================================
Wave Scheduling and Pipelining: A Complete Guide
========================================================

.. note::
   This document complements the first documentation written in ``wave_schedule.rst``.

Overview
========

This document explains Wave's pipelining system, detailing how schedules are created
and how they translate into the construction of pipelined loops.

**Two Main Concepts:**

1. **Initiation Interval (II):** The duration (in cycles) of one loop iteration
2. **Pipeline Stages:** The number of overlapping iterations running simultaneously (Pipeline Depth)

----

Part 1: Creating the Schedule
==============================

A schedule assigns precise timing information to every operation in the loop.
It answers two fundamental questions:

1. **Which Pipeline Stage?** (Determines data version/buffering)
2. **Which Cycle within the Stage?** (Determines execution order within II)

Example schedule structure:

.. code-block:: python

   schedule = {
       node1: {'stage': 0, 'cycle': 0},  # Runs in Stage 0, Cycle 0
       node2: {'stage': 0, 'cycle': 1},  # Runs in Stage 0, Cycle 1
       node3: {'stage': 1, 'cycle': 0},  # Runs in Stage 1, Cycle 0
   }

Manual Scheduling API
---------------------

The programmer explicitly controls staging using ``pl.set_stage()``.

   - Each call to ``pl.set_stage()`` creates a **New Pipeline Stage**
   - The list passed to the function defines the **Cycles within that stage** (determining the II)

Example: Creating a 2-Stage Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Call 1: Defines Stage 0 (Usually Global Loads / Producers)
   # This stage has 2 cycles (II=2).
   pl.set_stage([
       (global_load,),   # Cycle 0
       (),               # Cycle 1 (Empty gap for latency)
   ])

   # Call 2: Defines Stage 1 (Usually Compute / Consumers)
   # This stage MUST have the same II (2 cycles).
   pl.set_stage([
       (shared_load,),   # Cycle 0
       (mma,),           # Cycle 1
   ])



How Initiation Interval (II) is Determined
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The II is simply the **length of the list** passed to ``pl.set_stage()``.
All stages must have the same II (or compatible lengths) to align correctly.

.. code-block:: python

   # II = 3 (List has 3 entries)
   pl.set_stage([
       (op_A,),  # Cycle 0
       (op_B,),  # Cycle 1
       (),       # Cycle 2
   ])

----

Part 2: Loop Pipelining Construction
=====================================

Overview
--------

Once the schedule is set, the compiler constructs the pipelined loop.

The transformation occurs in ``loop_reconstruction.py``, which is responsible for:

- Taking the graph and schedule
- Generating **Prologue**, **Kernel**, and **Epilogue** nodes

Pipeline Structure
------------------

For a pipeline with **S** stages:

.. table::
   :widths: 20 80

   ================  =========================================================
   Phase             Description
   ================  =========================================================
   **Prologue**      Runs **S-1** iterations to fill the pipeline
   **Kernel**        The steady state where all stages run simultaneously
   **Epilogue**      Runs **S-1** iterations to drain the pipeline
   ================  =========================================================



How to Implement Triple Buffering (3 Stages) or more
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement **Triple Buffering**, you need **3 Stages**. The middle stage is
left empty to increase the pipeline depth between the Producer and Consumer.
By delaying the use of the data until the 3rd stage, we force the compiler to
keep the data alive longer, which triggers the allocation of a 3rd buffer.

.. code-block:: python

   # Stage 0: Load Data (Iter i+2)
   pl.set_stage([
       (global_load,),
       (),
   ])

   # Stage 1: Wait/Travel (Iter i+1)
   # Pure latency hiding stage.
   pl.set_stage([
       (),
       (),
   ])

   # Stage 2: Compute (Iter i)
   pl.set_stage([
       (shared_load,),
       (mma,),
   ])



Calculating Buffer Count
------------------------

The compiler determines buffer requirements by calculating the **Lifetime** of the data specifically,
the time from when the data is written to Shared Memory until the Local Load finishes reading it.

Key Terms:

:Producer: The Global Load operation (``global_to_shared``)
:Consumer: The Shared Memory Load operation (``shared_load``)
:Lifetime: ``Consumer_Start_Time - Producer_Start_Time``

The Buffer Count Formula
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{Buffer Count} = \lceil \frac{\text{Lifetime} + \text{OpDuration}}{II} \rceil

The formula accounts for the fact that the buffer must remain valid during the consumer's execution.

In our code, the duration of the local load is **1** , resulting in:

.. code-block:: python

   buffer_count = ceildiv(lifetime + 1, initiation_interval)

Example 1: 2-Stage Pipeline (Double Buffering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Stage 0: Producer (global_to_shared)
   pl.set_stage([
       (global_to_shared,),
       ()
   ])

   # Stage 1: Consumer (shared_load)
   pl.set_stage([
       (shared_load,),
       (mma,)
   ])

**Calculation:**

1. Producer Start: ``Stage 0 × II = 0 × 2 = 0``
2. Consumer Start: ``Stage 1 × II = 1 × 2 = 2``
3. Lifetime: ``2 - 0 = 2``
4. Buffer Count:

   .. math::

      \lceil \frac{2 + 1}{2} \rceil = \lceil 1.5 \rceil = 2 \text{ buffers}

Extending to 4+ Buffers
~~~~~~~~~~~~~~~~~~~~~~~~

To achieve **more than 3 buffers**, simply add extra empty stages between the producer and consumer.


Example: 4-Stage Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Stage 0: Producer
   pl.set_stage([(global_load,), ()])

   # Stage 1: Empty (Delay)
   pl.set_stage([(), ()])

   # Stage 2: Empty (Additional Delay)
   pl.set_stage([(), ()])

   # Stage 3: Consumer
   pl.set_stage([(shared_load,), (mma,)])

**Calculation:**

1. Producer Start: ``Stage 0 × II = 0 × 2 = 0``
2. Consumer Start: ``Stage 3 × II = 3 × 2 = 6``
3. Lifetime: ``6 - 0 = 6``
4. Buffer Count:

   .. math::

      \lceil \frac{6 + 1}{2} \rceil = \lceil 3.5 \rceil = 4 \text{ buffers }
