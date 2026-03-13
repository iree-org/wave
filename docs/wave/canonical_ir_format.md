# Canonical Region IR Format

This document describes the canonical region structure used by the Wave FX pipeline.

It is intentionally limited in scope: it only covers nested region interfaces
and capture structure. It does not attempt to define a full canonical form for
all FX graph details such as node ordering, `vector_shapes`, or write
dependencies.

## Goal

Wave is moving toward a single canonical region form so that:

- nested regions have one stable structural representation in Python
- FX <-> MLIR roundtrips preserve region structure in a form that later Python
  passes can continue to process
- for migration purposes, passes may declare which temporary non-canonical region view they need
  without forcing the whole pipeline to stay in that view

The canonical form is represented by `RegionFormat.ISOLATED` in
`wave_lang/kernel/wave/region_canonicalization.py`.

## Terms

- Outer source: a node defined outside a nested region but used by that region
- Local capture: the region-local representative of an outer source
- Capture signature: the ordered list stored on the parent `NestedRegionOp` in
  `implicit_captures`
- Direct outer reference: a region node operand that points straight to a node
  in the outer graph

## Canonical Form: `ISOLATED`

`ISOLATED` is the canonical/default region form.

Structural invariants:

- nested region bodies do not directly reference outer graph nodes
- every captured outer value used inside the region is represented by a region-local
  `Placeholder`
- these capture placeholders form the leading non-`IterArg` input prefix of the
  subgraph
- the parent `NestedRegionOp.implicit_captures` list is the authoritative
  ordered capture signature
- each local capture placeholder resolves to exactly one outer source via
  `meta["lifted"]`

In other words, the region interface is explicit and isolated from above.

Schematic shape:

```text
root graph:
  %outer_a
  %outer_b
  %region = iterate(..., implicit_captures=[%outer_a, %outer_b], ...)

region subgraph:
  %iter_arg0 = placeholder(iter arg)
  %outer_a = placeholder(lifted from outer)
  %outer_b = placeholder(lifted from outer)
  ...
```

## Temporary Non-Canonical Forms

Not all existing passes operate on `ISOLATED` yet. To migrate incrementally,
passes may request one of several temporary region views. The pass boundary
adapts into that form before the pass runs and, in the normal pipeline,
canonicalizes back to `ISOLATED` afterwards.

### `LEGACY_PLACEHOLDERS`

This is the older placeholder-based capture form still expected by some
pre-existing passes.

Structural properties:

- captured outer values are represented by region-local placeholders
- the mapping from a local placeholder back to its outer source may still be
  recovered with legacy heuristics, such as placeholder name or placeholder
  position within the capture prefix
- unlike `ISOLATED`, this form does not require `implicit_captures` plus
  `meta["lifted"]` to be the sole authoritative description of the capture
  interface

This is a weaker contract than `ISOLATED`. A pass marked
`LEGACY_PLACEHOLDERS` may still reason about captures through placeholder
layout or legacy lookup behavior instead of relying only on the canonical
capture interface. An already-canonical region may also satisfy this weaker
contract, so the adapter can be a no-op on some graphs.

This mode exists to support passes that still expect legacy placeholder
structure while the pipeline as a whole moves toward explicit canonical
captures.

### `DIRECT_OUTER_REF`

This is a legacy form where region bodies directly reference outer graph nodes.

Structural properties:

- operands inside the region may point directly to outer nodes
- capture placeholders are removed or bypassed where possible
- the parent capture signature may still track those outer values, but the body
  itself is not isolated from above

This form is convenient for passes that want to inspect or mutate the original
outer values directly, especially around captured memory operands.

### `SCHEDULE_SIGNATURE_PLACEHOLDERS`

This is a hybrid legacy form used by scheduling-related passes.

Structural properties:

- placeholders are kept only for schedule-signature sources
- non-signature captures are rewritten back to direct outer references
- the region mixes explicit placeholders for signature-defining values with
  direct outer references for everything else

In practice, the schedule-signature sources are the outer values that define
the region boundary from the scheduler's point of view, namely values such as
root placeholders and `NewRegister`s.

## Why There Are Multiple Forms

The long-term goal is for passes to converge on `ISOLATED`.

The intermediate forms exist because rewriting every pass at once would be too
large and too risky. Instead:

1. the pipeline keeps one canonical form
2. each pass declares the temporary form it currently expects
3. pass-boundary adapters convert into that form just for the duration of the
   pass
4. the normal pipeline returns to canonical form afterwards

This makes the migration incremental while still establishing one structural
source of truth.

## Pass Contract

Passes declare their required region form with `@requires_region_format(...)`.

The important contract is:

- if a pass does not declare a region form, it is assumed to operate on the
  canonical `ISOLATED` form
- in the normal pipeline, pass outputs are canonicalized back to `ISOLATED`
- white-box tests that want to inspect a temporary legacy form must request
  `canonicalize_output=False` explicitly at the call site

This keeps the default pipeline principled while still allowing tests to inspect
the raw intermediate structure when needed.

## Non-Goals

This document does not define:

- a canonical ordering for all FX nodes
- a canonical form for `GetResult` materialization beyond what is required by
  region structure
- a canonical form for `vector_shapes`, write dependencies, or other downstream
  analysis state

Those are separate concerns that may later build on top of this region-level
structural baseline.
