# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import sympy

from wave_lang.support.ir_imports import (
    gpu_d,
    VectorType,
    MemRefType,
    Context,
    IrType as Type,
    AffineMap,
    AffineMapAttr,
    IntegerAttr,
    Attribute,
)

if TYPE_CHECKING:
    from .kernel_model import KernelInfo, MemRefInfo


def parse_vector_type(s: str) -> Tuple[int, int, str]:
    """Parse vector type using MLIR Python bindings instead of string matching."""
    ctx = Context()
    try:
        vector_type = Type.parse(s, context=ctx)
        if not isinstance(vector_type, VectorType):
            raise ValueError(f"Expected VectorType, got {type(vector_type)}")

        return parse_vector_type_from_obj(vector_type)
    except Exception as e:
        raise ValueError(f"Failed to parse vector type '{s}': {e}")


def parse_memref_type(s: str) -> Tuple[List[int], List[int], int]:
    """
    Parse memref type using MLIR Python bindings instead of string matching.
    Parse e.g.: 'memref<16x16xf16, strided<[16, 1], offset: ?>>'
    Return (shape, strides_elems, elem_bytes)
    """
    ctx = Context()
    try:
        memref_type = Type.parse(s, context=ctx)
        if not isinstance(memref_type, MemRefType):
            raise ValueError(f"Expected MemRefType, got {type(memref_type)}")

        shape, strides_elems, elem_bytes = parse_memref_type_from_obj(memref_type)

        # Check if strides were explicitly provided (not defaulted)
        layout = memref_type.layout
        if not (hasattr(layout, "strides") and layout.strides):
            raise ValueError(f"Memref type '{s}' has no strides")

        return shape, strides_elems, elem_bytes
    except Exception as e:
        raise ValueError(f"Failed to parse memref type '{s}': {e}")


def parse_vector_type_from_obj(vector_type_obj) -> Tuple[int, int, str]:
    """Parse vector type directly from MLIR VectorType object."""
    if not isinstance(vector_type_obj, VectorType):
        raise ValueError(f"Expected VectorType, got {type(vector_type_obj)}")

    # Extract shape (first dimension for 1D vectors)
    assert len(vector_type_obj.shape) == 1, "Vector type must be 1D"
    n = vector_type_obj.shape[0] if vector_type_obj.shape else 1

    # Extract element type and bitwidth
    element_type = vector_type_obj.element_type
    eb = element_type.width // 8  # Convert bits to bytes

    # Get element type string representation
    et = str(element_type)

    return n, eb, et


def parse_memref_type_from_obj(memref_type_obj) -> Tuple[List[int], List[int], int]:
    """Parse memref type directly from MLIR MemRefType object."""
    if not isinstance(memref_type_obj, MemRefType):
        raise ValueError(f"Expected MemRefType, got {type(memref_type_obj)}")

    # Extract shape
    shape = list(memref_type_obj.shape)

    # Extract strides from layout
    layout = memref_type_obj.layout
    if hasattr(layout, "strides") and layout.strides:
        strides_elems = list(layout.strides)
    else:
        # Default to row-major strides if no explicit strides
        strides_elems = []
        stride = 1
        for dim in reversed(shape):
            strides_elems.insert(0, stride)
            stride *= dim

    # Extract element type bitwidth
    element_type = memref_type_obj.element_type
    elem_bytes = element_type.width // 8  # Convert bits to bytes

    return shape, strides_elems, elem_bytes


def attrs_to_dict(attrs) -> Dict[str, object]:
    return {x.name: x.attr for x in attrs}


def parse_wg_and_subgroup(
    attr_obj: Attribute,
) -> Tuple[Optional[Tuple[int, int, int]], Optional[int]]:
    """Parse workgroup and subgroup sizes from MLIR attribute using Python bindings."""
    wg = attr_obj.workgroup_size
    sg = attr_obj.subgroup_size
    return wg, sg


def tid_upper_bound_from_thread_id(op: gpu_d.ThreadIdOp) -> Optional[int]:
    """Extract upper bound from GPU thread ID operation using MLIR Python bindings."""
    # Check if the operation has an upper_bound attribute
    if hasattr(op, "upper_bound") and op.upper_bound is not None:
        if isinstance(op.upper_bound, IntegerAttr):
            return int(op.upper_bound.value)

    return None


def _apply_floor_division_pattern(
    expression: sympy.Expr, tid_upper_bound: int
) -> sympy.Expr:
    """Apply floor division simplification pattern: if x < N, then floor(x/N) = 0."""
    if tid_upper_bound is None or tid_upper_bound <= 0:
        return expression

    free_symbols = expression.free_symbols
    for symbol in free_symbols:
        symbol_name = str(symbol)
        if symbol_name == "tid_x":
            # Find all floor divisions involving tid_x and substitute with 0
            # when the divisor is greater than or equal to the upper bound
            floor_substitutions = {}

            # Find floor(x/N) patterns where N >= tid_upper_bound
            def find_floor_substitutions(expr):
                if isinstance(expr, sympy.floor):
                    # Check if this is floor(symbol/N) where N >= upper_bound
                    if len(expr.args) == 1:
                        arg = expr.args[0]
                        if isinstance(arg, sympy.Mul) and len(arg.args) == 2:
                            # Check if it's symbol * (1/N) or (1/N) * symbol
                            for i, factor in enumerate(arg.args):
                                if factor == symbol:
                                    # Get the other factor (should be 1/N)
                                    other_factor = arg.args[1 - i]

                                    # Handle Rational (like 1/64)
                                    if isinstance(other_factor, sympy.Rational):
                                        divisor = other_factor.denominator
                                        if int(divisor) >= tid_upper_bound:
                                            floor_substitutions[expr] = 0

                                    # Handle Pow with -1 exponent (like 1/N)
                                    elif (
                                        isinstance(other_factor, sympy.Pow)
                                        and other_factor.exp == -1
                                    ):
                                        divisor = other_factor.base
                                        if int(divisor) >= tid_upper_bound:
                                            floor_substitutions[expr] = 0
                elif hasattr(expr, "args"):
                    for arg in expr.args:
                        find_floor_substitutions(arg)

            find_floor_substitutions(expression)

            # Apply the floor substitutions
            if floor_substitutions:
                expression = expression.subs(floor_substitutions)

    return expression


def _apply_simplification_patterns(
    expression: sympy.Expr, tid_upper_bound: int
) -> sympy.Expr:
    """Apply a list of simplification patterns to the expression."""
    # List of pattern functions to apply
    pattern_functions = [
        _apply_floor_division_pattern,
        # Add more pattern functions here as needed
    ]

    # Apply each pattern function
    for pattern_function in pattern_functions:
        expression = pattern_function(expression, tid_upper_bound)

    return expression


def _create_dimension_substitutions(
    dimension_values: List,
) -> Dict[sympy.Symbol, sympy.Expr]:
    """Create substitutions for dimension values (d0, d1, ...)."""
    substitutions = {}

    for i, value in enumerate(dimension_values):
        dimension_symbol = sympy.Symbol(f"d{i}")
        if isinstance(value, int):
            substitutions[dimension_symbol] = sympy.Integer(value)
        elif value in ["tid.x", "tid.y", "tid.z"]:
            substitutions[dimension_symbol] = sympy.Symbol(
                value.replace(".", "_"), nonnegative=True
            )

    return substitutions


def _create_symbol_substitutions(symbol_values: List) -> Dict[sympy.Symbol, sympy.Expr]:
    """Create substitutions for symbol values (s0, s1, ...)."""
    substitutions = {}

    for i, value in enumerate(symbol_values):
        symbol_symbol = sympy.Symbol(f"s{i}")
        if isinstance(value, int):
            substitutions[symbol_symbol] = sympy.Integer(value)
        elif value in ["tid.x", "tid.y", "tid.z"]:
            substitutions[symbol_symbol] = sympy.Symbol(
                value.replace(".", "_"), nonnegative=True
            )

    return substitutions


def simplify_expression(
    map_attr: AffineMapAttr,
    tid_upper_bound: int,
    dim_values: List = None,
    symbol_values: List = None,
) -> Optional[sympy.Expr]:
    """Simplify affine map expression using SymPy and return the simplified expression."""
    if not isinstance(map_attr, AffineMapAttr):
        return None

    affine_map = map_attr.value
    if not isinstance(affine_map, AffineMap):
        return None

    # Get the result expressions
    results = affine_map.results
    if len(results) != 1:
        return None

    result_expr = results[0]

    # Convert MLIR affine expression to SymPy expression
    try:
        sympy_expr = _mlir_affine_to_sympy(result_expr)
        if sympy_expr is None:
            return None

        # Substitute dimension and symbol values if provided
        if dim_values is not None or symbol_values is not None:
            substitutions = {}

            # Create dimension substitutions
            if dim_values is not None:
                dimension_substitutions = _create_dimension_substitutions(dim_values)
                substitutions.update(dimension_substitutions)

            # Create symbol substitutions
            if symbol_values is not None:
                symbol_substitutions = _create_symbol_substitutions(symbol_values)
                substitutions.update(symbol_substitutions)

            # Apply substitutions
            if substitutions:
                sympy_expr = sympy_expr.subs(substitutions)

        # Apply pattern-based simplifications
        sympy_expr = _apply_simplification_patterns(sympy_expr, tid_upper_bound)

        # Return the simplified expression
        return sympy.simplify(sympy_expr)

    except Exception:
        # If SymPy conversion fails, return None
        return None


def _emit_constant_expression(
    expression: sympy.Expr, emitter, kernel_info, destination_register: str
) -> str:
    """Emit ASM for a constant expression."""
    from .instructions import VMovB32

    try:
        constant_value = int(expression)
        # Parse destination vreg index from string like "v2"
        assert destination_register.startswith("v")
        dst_v = int(destination_register[1:])
        emitter.emit_instruction(VMovB32(dst_v, constant_value))
        emitter.register_file.v_used.add(dst_v)
        return destination_register
    except:
        # If evaluation fails, throw an error
        raise ValueError(
            f"Cannot evaluate constant expression: {expression}. Expected a constant integer value."
        )


def _emit_thread_id_expression(
    expression: sympy.Expr, emitter, kernel_info, destination_register: str
) -> str:
    """
    Emit ASM for expressions involving thread ID symbols.

    Uses the generic ExprEmitter visitor to walk the expression tree and emit
    appropriate AMDGCN instructions.

    Supports:
    - Constants and tid_x
    - Additive chains (left-to-right accumulation)
    - Multiplication by integer (power-of-two -> shift, else mul)
    - Mod(expr, 2^k) -> VAndB32
    - floor(expr / 2^k) -> VLshrrevB32
    - Combinations of the above

    Raises ValueError on:
    - Non-power-of-two mod/div
    - Products of two dynamic sub-expressions
    """
    from .expression_emitter import ExprEmitter

    visitor = ExprEmitter(emitter, kernel_info)
    return visitor.emit(expression, destination_register)


def emit_expression_asm(expr: sympy.Expr, emitter, ki, dst_reg: str = "v2") -> str:
    """Emit ASM instructions for a simplified SymPy expression."""
    # Get all free symbols in the expression
    free_symbols = expr.free_symbols

    if len(free_symbols) == 0:
        # No symbols - this is a constant
        return _emit_constant_expression(expr, emitter, ki, dst_reg)

    # Handle expressions with thread ID symbols
    if len(free_symbols) == 1:
        symbol = list(free_symbols)[0]
        symbol_name = str(symbol)

        if symbol_name == "tid_x":
            # Handle expressions with tid_x (thread ID, nonnegative)
            return _emit_thread_id_expression(expr, emitter, ki, dst_reg)

    # If there are still symbols after simplification, we can't emit this directly
    # This means the expression depends on runtime values that we can't resolve
    # Throw an error instead of falling back
    free_symbols = expr.free_symbols
    symbol_names = [str(symbol) for symbol in free_symbols]
    raise ValueError(
        f"Cannot emit ASM for expression with unresolved symbols: {symbol_names}. Expression: {expr}"
    )


def _mlir_affine_to_sympy(expr) -> Optional[sympy.Expr]:
    """Convert MLIR affine expression to SymPy expression by parsing the string representation."""
    try:
        expr_str = str(expr)

        # Handle simple cases first
        if expr_str.isdigit():
            return sympy.Integer(int(expr_str))

        # Replace MLIR-specific operators with Python equivalents
        python_expr = (
            expr_str.replace("floordiv", "//")
            .replace("ceildiv", "ceil")
            .replace("mod", "%")
        )

        # Parse the expression using SymPy's parser
        from sympy.parsing.sympy_parser import parse_expr

        parsed_expr = parse_expr(python_expr)

        # Keep the original symbol names (d0, d1, s0, s1, etc.)
        # The caller will substitute actual values for these symbols
        return parsed_expr

    except Exception as e:
        raise ValueError(
            f"Failed to convert MLIR affine expression to SymPy expression: {e}"
        )


def build_memref_byte_offset_expr(
    indices_ssa: List[str], kernel_info: "KernelInfo", memref_info: "MemRefInfo"
) -> sympy.Expr:
    """
    Build a symbolic byte-offset expression from MLIR indices and memref strides.

    Computes: sum(index[i] * stride[i]) * elem_bytes

    Args:
        indices_ssa: List of SSA value strings representing indices
        kernel_info: Kernel information containing index environment
        memref_info: Memref information containing shape, strides, and element size

    Returns:
        SymPy expression representing the byte offset

    Raises:
        ValueError: If an index SSA is not found or has an unsupported type
    """
    # Mapping for thread ID strings to SymPy symbols
    TID_SYMBOLS = {
        "tid.x": sympy.Symbol("tid_x", nonnegative=True),
        "tid.y": sympy.Symbol("tid_y", nonnegative=True),
        "tid.z": sympy.Symbol("tid_z", nonnegative=True),
    }

    def to_sympy(value) -> sympy.Expr:
        """Convert index value to SymPy expression."""
        if isinstance(value, sympy.Expr):
            return value
        if isinstance(value, int):
            return sympy.Integer(value)
        if isinstance(value, str):
            if value in TID_SYMBOLS:
                return TID_SYMBOLS[value]
            raise ValueError(f"Unknown thread ID string: {value}")
        raise ValueError(f"Unsupported index type: {type(value)}")

    # Build element index expression: sum(index[i] * stride[i])
    elem_index_expr = sum(
        (to_sympy(kernel_info.index_env[idx_ssa]) * memref_info.strides_elems[dim])
        for dim, idx_ssa in enumerate(indices_ssa)
        if idx_ssa in kernel_info.index_env
    )

    # Verify all indices were found
    missing = [idx for idx in indices_ssa if idx not in kernel_info.index_env]
    if missing:
        raise ValueError(f"Index SSAs not found in kernel_info.index_env: {missing}")

    # Convert to byte offset
    return elem_index_expr * memref_info.elem_bytes


def split_const_dynamic(expr: sympy.Expr) -> Tuple[int, sympy.Expr]:
    """
    Split a SymPy expression into constant and dynamic parts.

    This is useful for GPU addressing modes that support base + voffset + instoffset,
    where instoffset is an immediate constant.

    Args:
        expr: SymPy expression to split (e.g., tid_x * 4 + 128)

    Returns:
        Tuple of (constant_term, dynamic_expr)
        - constant_term: Integer constant part (e.g., 128)
        - dynamic_expr: Remaining dynamic SymPy expression (e.g., tid_x * 4)

    Example:
        >>> split_const_dynamic(tid_x * 4 + 128)
        (128, tid_x * 4)
    """
    # Use SymPy's built-in as_coeff_Add which returns (const, rest)
    const, rest = expr.as_coeff_Add()

    # Convert constant to integer
    const_term = int(const) if const.is_number else 0

    # If rest is just the constant part, dynamic is zero
    dynamic_expr = rest if rest != const else sympy.Integer(0)

    return const_term, dynamic_expr


def build_element_byte_offset_exprs(
    value_vector_type,
    indices_ssa: List[str],
    kernel_info: "KernelInfo",
    memref_info: "MemRefInfo",
) -> List[sympy.Expr]:
    """
    Build per-element byte-offset expressions for a vector store.

    For a vector store of vector<NxT> to a memref, this computes the byte offset
    for each of the N elements by deriving everything from MLIR affine maps and indices.

    The MLIR indices already encode any complex layout patterns (e.g., MFMA layouts),
    so we just need to compute the base offset from the indices and add contiguous
    element offsets.

    Args:
        value_vector_type: The MLIR vector type being stored (e.g., vector<4xf32>)
        indices_ssa: List of SSA value strings representing store indices
        kernel_info: Kernel information containing index environment with affine maps
        memref_info: Memref information containing shape, strides, and element size

    Returns:
        List of SymPy expressions, one per vector element, representing byte offsets

    Raises:
        ValueError: If vector is not 1-D or indices are not found in environment
    """
    # Parse vector type to get number of elements
    num_elements, elem_bytes, _ = parse_vector_type_from_obj(value_vector_type)

    # Build base offset from MLIR indices (which include affine maps encoding layout)
    base_expr = build_memref_byte_offset_expr(indices_ssa, kernel_info, memref_info)

    # Determine element stride in bytes (innermost dimension)
    if len(memref_info.strides_elems) > 0:
        elem_stride_elems = memref_info.strides_elems[-1]
    else:
        elem_stride_elems = 1
    elem_stride_bytes = elem_stride_elems * memref_info.elem_bytes

    # Build list of expressions: base + k * elem_stride_bytes for k in 0..num_elements-1
    # The affine maps in indices_ssa already encode any complex access patterns
    exprs = []
    for k in range(num_elements):
        if k == 0:
            exprs.append(base_expr)
        else:
            offset_expr = base_expr + sympy.Integer(k * elem_stride_bytes)
            exprs.append(offset_expr)

    return exprs
