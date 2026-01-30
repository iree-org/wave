# Design for Elements Per Thread (EPT)

## Summary

**Recommendation:** Water does not need a separate `elements_per_thread` attribute or propagation pass. EPT is already encoded in index expressions as `index.size`. Only `wave.atomic` and `wave.scatter` require special attributes since they may process fewer elements than their input vector size.

**Note:** Water does not currently support `wave.atomic` or `wave.scatter`. The `atomic_count` and `scatter_count` attributes will be considered when support for these ops is introduced.

---

## Background:

EPT represents how many data elements each thread processes. It determines:
- Vector width for memory operations (Read/Write -> `vector<EPT x f32>`)
- Loop unrolling for scalar ops (Atomic, Scatter)

---

## How PyWave Handles EPT

### Index Expressions

Every operation has an index expression per dimension:
```
IndexSequence(start, size, stride)
```

Where:
- `start` = where this thread begins (e.g., `thread_id * EPT`)
- `size` = **elements per thread**
- `stride` = memory stride between elements

### Propagation in PyWave

1. **Anchor ops** (Read, Write, MMA, Reduce) compute their index with `size`
2. `propagate_indices()` spreads index expressions to all connected ops
3. After propagation, **every op has index with correct `size`**

---

## Detailed Per-Operation Analysis

### Read/Write

| Aspect | Details |
|--------|---------|
| **Has EPT attribute** | Yes |
| **EPT source** | User-specified or computed from `tile_size / threads_per_block` |
| **How vector size is defined** | `apply_read_write_thread_mapping()` creates `IndexSequence` with `size` = EPT |
| **Lowering uses** | `elements_per_thread` from `node.args` (redundant with `index.size`) |
| **EPT needed in Water?** | No |

### MMA

| Aspect | Details |
|--------|---------|
| **Has EPT attribute** | No |
| **EPT source** | MMA intrinsic layout (e.g., 4 for F32_16x16x16_F16 ACC) |
| **How vector size is defined** | `apply_mma_mapping()` creates `IndexSequence` with intrinsic-defined `size` |
| **Lowering uses** | Vector types inherited from operands |
| **EPT needed in Water?** | No |

### Reduce

| Aspect | Details |
|--------|---------|
| **Has EPT attribute** | No |
| **How vector size is defined** | Creates `IndexSequence` with computed `size` = `vector_shapes[dim] / threads_per_wave` |
| **Lowering uses** | Vector types inherited from operands |
| **EPT needed in Water?** | No |

### Broadcast

| Aspect | Details |
|--------|---------|
| **Has EPT attribute** | No |
| **How vector size is defined** | `index.size` inherited from source via propagation |
| **Lowering uses** | `index.size` directly (`handlers.py:1924`) |
| **EPT needed in Water?** | No |

### Atomic

| Aspect | Details |
|--------|---------|
| **Has EPT attribute** | Yes |
| **EPT source** | User-specified, or `index.size` if None |
| **How EPT is used ** | Controls loop unrolling |
| **Lowering uses** | `elements_per_thread` from `node.args` |
| **EPT needed in Water?** | Yes since it can differ from vector size |

Real use case where EPT differs from vector size (`tests/kernel/e2e/test_atomic.py:238-244`):
```python
one_vec = tkw.Register[NUM_EXPERTS, dtype](1)  # 4 elements
tkw.atomic_add(
    one_vec,
    shmem,
    mapping=histogram_read,
    elements_per_thread=1,  # Only process 1 element, not 4!
)
```

### Scatter

| Aspect | Details |
|--------|---------|
| **Has EPT attribute** | Yes (default=1) |
| **EPT source** | User-specified |
| **How EPT is used ** | Controls loop unrolling |
| **Lowering uses** | `elements_per_thread` from `node.args` |
| **EPT needed in Water?** | Yes since it can differ from vector size |

### Binary/Unary Ops (add, mul, exp2, etc.)

| Aspect | Details |
|--------|---------|
| **Has EPT attribute** | No |
| **How vector size is defined** | Vector types from operands |
| **Lowering uses** | Vector types inherited from operands |
| **EPT needed in Water?** | No |

---

### Key Insight

The explicit `elements_per_thread` attribute on Read/Write/Atomic is **redundant**:
```python
# In populate_read_write_source_indices:
index[dim] = IndexSequence(
    thread_id * elements_per_thread,  # start
    elements_per_thread,              # size  <-- EPT is here
    stride
)
```

EPT is stored in two places (attribute and index.size) but they're the same value.

---

## Proposed Water Design

### Principle

**EPT = `index.size`**. No separate concept needed.

### Design

1. **No `elements_per_thread` attribute** on any operation except Atomic and Scatter
2. **Index expression propagation** handles EPT propagation (size is one field)
3. **Lowering** reads `index.size` to determine vector width
4. **Atomic** has special `atomic_count` attribute (may differ from source's index.size)
5. **Scatter** has special `scatter_count` attribute (may differ from source's index.size)

### Why Atomic and Scatter are Special

Atomic and Scatter may intentionally process fewer elements than the source register holds:
```
Source register: vector<4xf32> (index.size = 4)
Atomic/Scatter count: 1 (only process first element)
```

### Operation Handling

| Operation | Water Handling |
|-----------|----------------|
| **Read** | Lowering reads `index.size` -> `vector.transfer_read` with that width |
| **Write** | Lowering reads `index.size` -> `vector.transfer_write` with that width |
| **MMA** | `index.size` from MMA intrinsic constraints |
| **Reduce** | Input/output `index.size` determines reduction pattern |
| **Broadcast** | Source/target `index.size` determines broadcast pattern |
| **Atomic** | `atomic_count` attribute (not yet supported) |
| **Scatter** | `scatter_count` attribute (not yet supported) |
| **Binary/Unary** | `index.size` inherited, determines vector width |

---

## Migration Steps

1. **Update lowering patterns** to read `index.size` instead of expecting pre-converted vector types
2. **Update verification** to check index.size consistency
3. **Remove `elements_per_thread` attribute** from Read, Write, MMA, Reduce, Broadcast op definitions
4. **Simplify/remove `PropagateElementsPerThread` pass** - subsume into index propagation
5. **Add `atomic_count` attribute** to Atomic op when support is introduced
6. **Add `scatter_count` attribute** to Scatter op when support is introduced

---

## Example

**Current:**
```mlir
%0 = wave.read %mem {elements_per_thread = 4, index = {...}}
    : !wave.tensor<[@M] of f32, <global>> -> !wave.tensor<[@M] of f32, <register>>

// After PropagateElementsPerThread pass:
%0 = wave.read %mem {...} : !wave.tensor<...> -> vector<4xf32>
```

**Proposition:**
```mlir
%0 = wave.read %mem {index = {M: (tid*4, 4, 1)}}  // size=4 is the EPT
    : !wave.tensor<[@M] of f32, <global>> -> !wave.tensor<[@M] of f32, <register>>

// Lowering directly creates:
%0 = vector.transfer_read %mem[...] : memref<...> -> vector<4xf32>
```

---
