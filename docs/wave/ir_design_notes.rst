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


Index Expressions
-----------------

Index expressions describe how elements of a tensor are distributed across
multiple workitems (threads) and workgroups (blocks), as well as across multiple
copies of the operation processing the tensor after the _expansion_ (unrolling)
process. They are recorded per dimension and consist of three components:

  - `start` (also `offset`) - position of the first element "owned" by the
    workitem along this dimension;
  - `step` (also `size`) - number of elements "owned" by the workitem along this
    dimension;
  - `stride` - number of elements in linear indexing of this dimension or,
    **inconsistently**, of the canonical row-major storage needed to get to the
    next element "owned" by the workitem.

Note the inconsistency in stride definition, which depends on the kind of
operation that the index expression was propagate from: per-dimension strides if
propagated from MMA-style operations, row-major Torch-style strides otherwise.

In Python, each component of the index expression is a sympy `Expr` using
specially-named symbols to refer to workgroup and item identifiers.
Per-dimension expressions are organized in a dictionary where their order is
load-bearing and matches that of the `indexing_dims` property of the parent
operation. Typically, this order is that of unique non-reduction dimensions of
the consecutive operands followed by reduction dimensions. For memory access
operations, the order is that of the non-memory operand. Index expressions are
associated with the operation via the `index` attribute, which may contain
multiple entries; one entry typically corresponds to the operation's result;
multiple entries typically correspond to the operation's operands.

In MLIR, each component of the index expression is an `AffineMap` using
exclusively symbols. The entirety of the components uses the same list of
symbols, which is associated with a list of custom attributes for each kind of
special value, sorted upon construction. Per-dimension expressions are organized
into a dictionary attribute and their order is captured in a separate attribute
called `ordered_syms`, which needs to be revised. Index expressions are lattice
objects attached to a value by a dataflow analysis, but they may be attached to
operations similarly to Python for verification purposes.

**Both** implement a heuristic that avoids propagating index expressions if they
don't cover non-unit dimensions of the vector shape, all dimensions if the rank
of the index expression dictionary is larger, or a subset thereof if it is
smaller (but the dictionary must not contain other expressions). This heuristic
is based on some notion of "completeness" of the index expressions, but is
practically needed to avoid conflicts in attention kernels. A more robust
solution would require comprehensive re-layout insertion.

Index Sequence analysis
-----------------------

Index sequence analysis determines index expressions for relevant objects.

**In Python**, index "sequences" are associated with an operation and are
typically reflective of the operation's result. For MMAs, the sequence contains
piecewise expressions with the condition indicating whether the piece
corresponds to the accumulator *operand* (or identically the result) or the
LHS/RHS operands.

The process is driven by a naive pseudo-dataflow analysis that traverses the
transitive sources of operation's operands (backward slice) and the transitive
users of operation's results (forward slice) in a greedy manner. Depending on
the presence of MMA operations or Reduction operations, different logic is used
to determine the index expressions with no fallback. That is, if the MMA
operations are present, the logic specific to Reduction operations is never
triggered. This may lead to operations missing index expressions.

**in MLIR**, index expressions are a property of a *value* indicating how it is
distributed across workgroups, workitems and copies of the operation (instances)
that appear after the expansion process. For compatibility, they may be
reflected in the ``index`` attribute of an operation.

The process is a combined forward and backward dataflow analysis sharing a
lattice. To reflect Python behavior, it is using a priority mechanism based on
the "source" of the index expression -- MMA, Reduction or Write, in order --
with higher-priority expressions overriding lower-priority ones instead of
causing propagation conflicts. Furthermore, MMA operations have additional
priority between themselves following the walk order in IR post-order traversal,
earlier MMA operations have higher priority.

This complexity is entirely due to backwards compatibility with Python and
should be entirely reworked towards a sound and consistent layout propagation
dataflow analysis.


IndexMapping
------------

The `mapping` attribute on read and write operations on the Python side allows
for separate mappings for "inputs" (memory operand for reads, value operand for
writes) and "outputs" (value operand for reads, memory operand for writes).
Mnemonically, "inputs" means where the data currently resides and "outputs"
means where it will be after the operation. Each of these is a dictionary from
symbol names used to index the corresponding tensor to a full-fledged sympy
expression that may involve, in addition to the usual symbols, special
placeholder `iterator` symbols that refer to positionally-indexed "iterators".
Note that "iterators" do **not** correspond to the surrounding iteration space
as defined by `tkw.iterate` and the grid configuration, they are mere positional
identifiers. There is a soft, undocumented assumption that only the mapping that
corresponds to the memory operand is allowed to have a non-identity mapping, but
it is not verified directly. The order of elements in this dictionary is
load-bearing, though its exact meaning is not properly documented. It does not
necessarily match the order of symbols in the shape. The other mapping, for
non-memory, is soft-expected to be identity, i.e., the symbols are mapped to
positional iterators where the position matches the position of the symbol in
the corresponding shape. None of this has verification logic and unsupported
cases just hit assertions or other exceptions inside the compilation flow.

Practically, these mappings live in relational spaces `shape symbols ->
"iterators"` which are inverted and composed to effectively obtain the mapping
`shape symbols -> shape symbols` needed to correlate locations of elements
between two tensors. The soft requirement for one of the mappings to be identity
allows for it to be inverted. Since it is always the non-memory mapping, it
becomes possible to define the `memory shape -> value shape` mapping by
composition. The implementation tries a direct construction of the inverse by
mapping positional "iterators" to shape symbols and checks that the same
"iterator" is used in RHS for the same symbol in both mappings.

MLIR representation captures directly the `memory shape -> value shape` mapping
as an `AffineMap`, currently restricted to permutations only but with a
possibility of future relaxation. It uses `AffineMap` dimensions as positional
identifiers and thus avoids the need to introduce any special "iterator" symbols
and the related naming confusion.

In Python, mappings are taken into account primarily during _code generation_ of
reads and writes: index expressions that appear in a specific order are mapped
to positional iterators with the same position. Then this mapping is used to
update the mapping from the memory shape dimensions to co-indexed iterators,
potentially resulting in a permuted index expression list. For example, given a
memory shape `[A, B, C, D]` and a mapping `{A: i0, B: i3, C: i2, D: i1}` first
creates a map `{i0: index[A], i1: index[B], i2: index[C], i3: index[D]}` and
then obtains the permuted index map `{A: index[A], B: index[D], C: index[C], D:
index[B]}`.

Conceptually, the index mapping may be seen as a function in vector space that
is (pre-)composed with index expressions interpreted as another function in
vector space. The signature of the index expression function would be
`workgroup/item -> value address space` and that of the mapping would be `memory
address space -> value address space`, requiring the latter to be inverted
before composition as described above.

In Python, only the `start` components of the index expressions are composed,
presumably incorrectly, because the other components are no longer needed at the
code generation stage and because of the unclear effect a mapping with offsets
would have on components that are not offsets. In MLIR, all components are
permuted. In practice, permutation means a change to the `ordered_syms`
attribute without changing the dictionary itself.
