IR Design Notes
===============

This document records design decisions, conventions, and known subtleties in
Wave's internal representations (FX graph and MLIR).  It is intended as a
reference for contributors working on the FX <-> MLIR conversion layer and the
graph comparison infrastructure.


FX Node Attributes: Dataclass Fields vs. Dynamic Attributes
------------------------------------------------------------

Wave operations are modelled as Python `dataclass`es that wrap
`torch.fx.Node`.  The node carries two kinds of semantic state:

1. **Dataclass fields** – declared on the `CustomOp` subclass (e.g.
   `MMA.lhs`, `Read.memory`).  These are discovered via
   `dataclasses.fields()` and compared automatically by the graph
   equivalence checker.

2. **Dynamic attributes** – set on the `fx.Node` object at runtime via
   `setattr` (e.g. `node.index`, `node.vector_shapes`,
   `node.reduction_dim`).  Because they are not dataclass fields, the
   comparison logic in `graph_utils._check_nodes_equivalent` must
   enumerate them explicitly (see `_ADDITIONAL_NODE_ATTRS`).

A third category of dynamic attributes is *non-semantic* and deliberately
excluded from comparison: `location`, `expanded_dims`, and
`scheduling_parameters`.  These are artefacts of scheduling or debugging
and do not affect functional equivalence.


MMA Index Representation
------------------------

**MLIR side** – The `index` attribute on `wave.mma` is an `ArrayAttr`
with exactly **four** entries: one per operand (lhs, rhs, acc) plus one for
the result.  This matches `MmaOp::setIndexFromLattices` which serialises
`operandExprs + resultExprs`.  The result entry is always identical to the
accumulator entry because the MMA result type equals the accumulator type.

**Python/FX side** – A single `dict[IndexSymbol, IndexSequence]` using
`sympy.Piecewise` to encode the `MMA_ACC` conditional (e.g. "for the M
dimension: use the LHS index when not accumulator, use the ACC index when
accumulator").

The FX -> MLIR emitter (`water_emitter.py`) decomposes the merged
Piecewise dict into four per-value entries.  The MLIR -> FX converter
(`attr_type_converter.convert_index_mapping_array_to_sympy`) reads the
four entries and reconstructs the Piecewise form.


Implicit Captures in Iterate Subgraphs
---------------------------------------

On the MLIR side, the `IterateOp` body region supports both forms:
captures can appear as explicit block arguments (`IsolatedFromAbove`),
or the region can reference values from the enclosing scope directly.
The `makeIsolated` / `makeNonIsolated` transformations switch between
them.  On the FX side, two analogous representations coexist:

- **Lifted placeholders** – The subgraph contains `Placeholder` nodes
  with `node.meta["lifted"]` pointing to the outer node.  This form
  appears after tracing and before the hoisting pass.

- **Direct references** – The subgraph references outer `fx.Node` values
  directly, without intermediate placeholders.  This form appears after the
  hoisting pass eliminates the lifted placeholders.

Which form a traced kernel ends up with depends on the pipeline and
scheduling passes that have run on it.  For example, in the FX emitter
roundtrip tests the simple matmul kernel uses lifted placeholders while
the pipelined GEMM kernel uses direct references.  Both are valid output
of the Wave compiler.

The MLIR -> FX converter produces direct references (mapping capture block
arguments to outer values).  The graph comparison infrastructure
(`graph_utils._reconcile_lifted_placeholders`) handles both forms so
that traces can be compared regardless of which representation they use.

Note that the `implicit_captures` field on `Iterate` / `Conditional`
is **not a reliable source of truth** after hoisting.  When the hoisting
pass erases a lifted placeholder, `Placeholder.erase()` runs a
dead-capture cleanup that removes the corresponding entry from
`implicit_captures` — even if the subgraph still references the outer
node directly.  The MLIR side does not have this problem: `makeIsolated`
walks the region and discovers all outer references regardless of how they
are represented.  As a result, the MLIR-imported trace may list more
captures than the source trace.


IndexMapping
------------

The `mapping` attribute on read and write operations on the Python side allows
for separate mappings for "inputs" (memory operand for reads, value operand for
writes) and "outputs" (value operand for reads, memory operand for writes). Each
of this is a dictionary from symbol names used to index the corresponding tensor
to a full-fledged sympy expression that may involve, in addition to the usual
symbols, special placeholder `iterator` symbols that refer to
positionally-indexed iterators of the notional iteration space that surrounds
the op. The order of elements in the dictionary is load-bearing, though its
exact meaning is not properly documented. It does not necessarily match the
order of symbols in the shape. None of this has verification logic and
unsupported cases just hit assertions or other exceptions inside the compilation
flow.

The simultaneous presence of both "inputs" and "outputs" mapping means that one
of them may be kept as identity, i.e., the symbols are mapped to positional
iterators where the position matches the position of the symbol in the
corresponding shape. For reads, this is the "outputs" mapping and, for writes,
this is the "inputs" mapping. There is currently no enforcement that it is
indeed the case, only a verbalized implicit assumption. This redundancy allows
one to (almost always) map every symbol to a single positional iterator. When a
more complex expression is used, additional logic attempts to extract the single
iterator that is used in it. This in turn allows to compute a permutation of
dimensions during _code generation_ of reads and writes: index expressions that
appear in a specific order are mapped to positional iterations with the same
position. Then this mapping is used to update the mapping from the memory shape
dimensions to co-indexed iterators, potentially resulting in a permuted index
expression list. For example, given a memory shape `[A, B, C, D]` and a mapping
`{A: i0, B: i3, C: i2, D: i1}` first creates a map `{i0: index[A], i1: index[B],
i2: index[C], i3: index[D]}` and then obtains the permuted index map
`{A: index[A], B: index[D], C: index[C], D: index[B]}`.
