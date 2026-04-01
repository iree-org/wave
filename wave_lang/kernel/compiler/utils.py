from math import prod
from typing import Any, Optional

from .._support.indexing import IndexingContext, IndexSymbol


def symbolic_strides_match_physical_memory(memory: Any, symbolic_shape: tuple) -> bool:
    """Return True when dense strides from *symbolic_shape* match the buffer layout.

    Memories with an explicit ``physical_layout`` whose shape differs from the
    symbolic shape have memref strides that do not match
    :func:`strides_from_symbolic_shape`.  Linearizing read indices with the wrong
    implied strides is incorrect; callers must skip flattening in that case.

    ``MemoryLayout`` (physical_layout) stores only a shape; strides are always
    derived as row-major from that shape.  Therefore shape equality implies
    stride equality -- no explicit stride comparison is needed.
    """
    mem_type = getattr(memory, "type", None)
    if mem_type is None:
        return True
    layout = getattr(mem_type, "physical_layout", None)
    if layout is None:
        return True
    layout_shape = layout.shape
    if len(layout_shape) != len(symbolic_shape):
        return False
    return all(l == s for l, s in zip(layout_shape, symbolic_shape))


def strides_from_symbolic_shape(
    indexing_context: IndexingContext,
    symbolic_shape: Optional[list[IndexSymbol]],
    allow_mixed_shapes: bool = False,
) -> Optional[list[int]]:
    """
    Computes the stride from a given symbolic shape and indexing context,
    assuming the innermost dimension is contiguous.
    """
    if symbolic_shape is None:
        return None
    static_shape = [indexing_context.get_static_value(sym) for sym in symbolic_shape]
    if None in static_shape and not allow_mixed_shapes:
        return None
    mixed_shape = [
        static if static is not None else dynamic
        for static, dynamic in zip(static_shape, symbolic_shape)
    ]
    strides = []
    for i in range(1, len(mixed_shape)):
        strides.append(prod(mixed_shape[-i:]))
    return strides[::-1] + [1]
