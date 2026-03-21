# Design: Index Flattening Pass (`flatten_read_indices`)

## Goal

Convert every Read op from an N-D logical index to a 1-D physical index.
The pass runs after `generate_bounds_exprs` and before `merge_contiguous_reads`.
After the pass, every Read op's address is described by a single linearised
offset, making downstream contiguity analysis and IV stride annotation trivial.

## Current State

A Read op carries three index-related fields:

```
index:    dict[IndexSymbol, IndexSequence]   e.g. {M: IS(m_start, 4, 1), K: IS(k_start, 1, 1)}
mapping:  Optional[IndexMapping]             logical -> physical coordinate transform
bounds:   Optional[dict[IndexSymbol, IndexExpr]]  e.g. {M: M_BOUND}
```

Codegen (`handle_read` in `read_write.py`) consumes these as follows:

1. If `mapping` is present, apply it to `index` to get `transformed_index`
   (physical coordinates), then use `transformed_index` for address computation
   and pick either transformed or logical index for masking.
2. `_build_start_indices(index)` iterates the dict, producing one MLIR index
   per dimension.
3. `_linearize_memref` dots these per-dim indices with physical memref strides
   to get a single flat offset.
4. `_build_mask` / `_build_mask_with_mapping` evaluates `index[dim] < bounds[dim]`
   per bounded dimension.

## Target State

Introduce a special symbol `LINEAR_INDEX` (defined in `global_symbols.py`).
After the flattening pass, a Read op looks like:

```
index:    {LINEAR_INDEX: IndexSequence(flat_offset, ept, 1)}
mapping:  None                               (merged into index)
bounds:   None or dict[IndexExpr, IndexExpr] (expression-keyed, see "Bounds")
```

A single-entry dict with `LINEAR_INDEX` as the key signals to codegen that
the address is already a flat physical offset -- skip per-dim linearisation
and use the offset directly.

## Transformation Steps

For each Read op:

### 1. Compute physical starts

| Case | Action |
|------|--------|
| `mapping is None` or identity | Physical starts = `index[dim].start` for each dim. |
| Non-identity mapping, no dynamic vals | Apply `transform_index_on_mapping` to get physical starts. |
| Non-identity mapping WITH dynamic vals | Skip (see "Dynamic Mappings"). |

### 2. Linearize

Compute strides from the memory's physical shape.  For shared memory with
a `distributed_shape` (padded for bank-conflict avoidance), use that instead
of `symbolic_shape` -- this matches the codegen logic in `_create_vec_read_write`
(`read_write.py:598-601`).

```python
memory = get_custom(custom.memory)
if memory.type.address_space == SHARED_ADDRESS_SPACE and hasattr(memory, "distributed_shape"):
    shape = memory.distributed_shape
else:
    shape = memory.type.symbolic_shape

strides = strides_from_symbolic_shape(idxc, shape, allow_mixed_shapes=True)
flat_start = sum(phys_start[dim] * stride for dim, stride in zip(dims, strides))
```

### 3. Convert bounds to expression-keyed form

Expand bounds semantics from `{dim_symbol: bound}` to `{dim_expr: bound}`.
The key is a sympy expression that, when less than the bound, means the
access is in-range.

```python
idxc = IndexingContext.current()
flat_with_iota = flat_start + idxc.iota(ept)
shape_sizes = [subs_idxc(d) for d in symbolic_shape]
coords = delinearize_index(flat_with_iota, shape_sizes)
# coords[i] is the delinearised coordinate for symbolic_dims[i]
# e.g. coords = [(flat+iota) // K_SIZE, (flat+iota) % K_SIZE]

new_bounds = {}
for dim, bound in original_bounds.items():
    if dim not in symbolic_dims:
        continue  # skip bounds on dims not in this memory's shape
    dim_idx = symbolic_dims.index(dim)
    new_bounds[coords[dim_idx]] = bound
# e.g. {(flat+iota) // K_SIZE: M_BOUND, (flat+iota) % K_SIZE: K_BOUND}
```

The mask condition is then `And(expr < bound for expr, bound in new_bounds.items())`,
which is exactly the original `m_coord < M_BOUND AND k_coord < K_BOUND` but
expressed in terms of the flat offset.  Iota is already in the key expressions,
so `_build_mask` does not need to add it separately.

### 4. Rewrite the op

```python
custom.index = {LINEAR_INDEX: IndexSequence(flat_start, ept, 1)}
custom.update_arg("mapping", None)
custom.update_arg("bounds", new_bounds)   # expression-keyed or None
```

Also clear `mapping_dynamic_vals` if the mapping was resolved.

## `LINEAR_INDEX` Symbol

Defined in `global_symbols.py`:

```python
LINEAR_INDEX = index_symbol("LINEAR_INDEX")
```

### Detection helpers (in `general_utils.py`)

```python
def is_flattened_index(index: dict) -> bool:
    return LINEAR_INDEX in index

def get_flat_offset(index: dict) -> IndexExpr:
    return index[LINEAR_INDEX].start
```

All consumers use these instead of checking `LINEAR_INDEX in custom.index`
directly.

### Codegen changes (`handle_read` / `handle_write`)

When `LINEAR_INDEX` is in the index:

1. **Address**: use `index[LINEAR_INDEX].start` as the flat byte offset
   directly.  Reinterpret the memref as 1-D (the IV-split fast path at
   lines 996-1049 already does this via `_linearize_memref` with zero
   indices), then `vector.load(lin_src, [flat_offset])`.

2. **Mask**: `_build_mask` handles expression-keyed bounds (see below).
   No mapping, no `_build_mask_with_mapping`.

3. **No mapping application**: mapping is None, so the codegen mapping
   branch (lines 975-989) is skipped entirely.

### merge_contiguous_reads changes

After flattening, reads have `LINEAR_INDEX` as the only key.  The merge
pass needs to detect this and treat `index[LINEAR_INDEX].start` as the
flat offset directly -- no `_get_physical_start`, no stride computation.
The merge dim is always `LINEAR_INDEX`.

## Bounds: Expression-Keyed Semantics

### Current semantics

```python
bounds = {M: M_BOUND}       # means: index[M].start < M_BOUND
```

`_build_mask` looks up `index[dim].start`, adds iota to the fastest dim,
then checks `start < bound`.

### New semantics (backward-compatible extension)

```python
bounds = {expr: bound}       # means: expr < bound
```

When the key is an `IndexSymbol` present in `index`, `_build_mask` behaves
exactly as before (look up start, add iota to fastest dim).  When the key is
a general `IndexExpr`, the expression IS the value to compare -- iota is
already embedded if needed.

### Change to `_build_mask`

```python
def _build_mask(emitter, index, elements_per_thread, bounds, ...):
    if not bounds:
        return None
    idxc = IndexingContext.current()
    conditions = []
    for key, bound in bounds.items():
        if isinstance(key, IndexSymbol) and key in index:
            # Legacy per-dim bound.
            start = _get_start_index(index[key])
            if key == list(index)[get_fastest_index(index)]:
                start = start + idxc.iota(elements_per_thread)
            conditions.append(start < bound)
        else:
            # Expression-keyed bound (from flattened index).
            # Iota already embedded in key expression.
            conditions.append(key < bound)
    mask_expr = reduce(And, conditions)
    mask = gen_sympy_index(add_emitter_subs(emitter, dynamic_values), mask_expr)
    ...
```

### Dynamic shape dims

`delinearize_index` uses `Mod` and `floor` with shape sizes as denominators.
Static sizes produce clean expressions: `(flat + iota) % 128`.  Symbolic
sizes produce `Mod(flat + iota, DYN_K)` and `floor(flat / DYN_K)` -- these
are valid sympy expressions that `gen_sympy_index` lowers to
`arith.remui` / `arith.divui`.  Dynamic shapes work correctly.

## Dynamic Mappings

Reads with `mapping_dynamic_vals` are **skipped** by the flattening pass.
The existing codegen path handles them correctly, and `merge_contiguous_reads`
already skips them too (line 850).  Merging and IV extraction won't run for
these ops but that's acceptable -- they're uncommon and the existing lowering
works.  Proper handling of dynamic mapping vals needs more thought and is
deferred.

## What This Enables

1. **merge_contiguous_reads simplification.**  After flattening, the flat
   offset is `index[LINEAR_INDEX].start`.  The merge pass no longer needs
   `_get_physical_start`, `_get_read_transform_info`, stride computation,
   or `transform_index_on_mapping`.  Contiguity is just
   `offset_a + ept == offset_b`.

2. **annotate_iv_strides pass (step 3 from Hardcode84's comment).**
   With a flat index, IV stride extraction is:
   ```python
   stride = simplify(flat_index.subs(iv, iv + step) - flat_index)
   ```
   Producing `START + IV * STRIDE` form.  This is consumed by waveasm's
   `BufferLoadStrengthReduction` which already does exactly this at the
   assembly level.

3. **Removal of codegen's IV-split fast path.**  The `_try_iv_split_offset`
   logic in `handle_read` (lines 996-1049) does ad-hoc linearisation and
   IV hoisting.  With pre-linearised indices and annotated strides, this
   becomes unnecessary.

## Scope for Step 1

- Define `LINEAR_INDEX` in `global_symbols.py`.
- Implement `flatten_read_indices` pass:
  - Flatten unmapped / identity-mapped reads.
  - Flatten non-identity mapped reads WITHOUT dynamic vals.
  - Skip reads with dynamic mapping vals (option A).
  - Convert bounds to expression-keyed form using `delinearize_index`.
- Wire into `compile.py` between `generate_bounds_exprs` and
  `merge_contiguous_reads`.
- Update `_build_mask` to handle expression-keyed bounds.
- Update `handle_read` / `handle_write` codegen to detect `LINEAR_INDEX`
  and use the flat offset directly.
- Update `merge_contiguous_reads` to detect `LINEAR_INDEX` and use
  `index[LINEAR_INDEX].start` as the flat offset.

## Interactions With Other Passes

| Pass | Runs | Impact |
|------|------|--------|
| `simplify_indices` | Before | No impact, indices are still N-D. |
| `partition_gather_like_ops` | Before | No impact, needs N-D structure. |
| `generate_bounds_exprs` | Before | Sets `bounds` which we convert. |
| `merge_contiguous_reads` | After | Detect LINEAR_INDEX, use flat offset directly. |
| `handle_read` codegen | After | Detect LINEAR_INDEX, skip linearisation, use flat offset. `_build_mask` handles expr-keyed bounds. |

## Shared Memory Reads

Shared memory uses the same row-major linearization as global memory, but
the strides come from `distributed_shape` (which may include padding for
bank-conflict avoidance) rather than `symbolic_shape`.

In codegen, `_linearize_shared_mem` reinterprets the ND memref as 1-D,
then `linearize_index(node_index, stride_values)` computes the flat offset
(`read_write.py:648-653`).  This is the same `sum(start * stride)` formula.

The flattening pass handles shared memory by using `distributed_shape` for
stride computation when present (see step 2).  For `LINEAR_INDEX` reads,
codegen reinterprets the memref as 1-D and uses the pre-computed flat offset
directly -- the same end result, just computed earlier.

## Open Questions

1. Should we also flatten Write ops?  Writes could benefit for IV stride
   annotation.  Punt to a follow-up.
