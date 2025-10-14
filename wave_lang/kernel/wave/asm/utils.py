# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Optional, Tuple

import sympy as sp

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


def _apply_floor_division_pattern(expression: sp.Expr, tid_upper_bound: int) -> sp.Expr:
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
                if isinstance(expr, sp.floor):
                    # Check if this is floor(symbol/N) where N >= upper_bound
                    if len(expr.args) == 1:
                        arg = expr.args[0]
                        if isinstance(arg, sp.Mul) and len(arg.args) == 2:
                            # Check if it's symbol * (1/N) or (1/N) * symbol
                            for i, factor in enumerate(arg.args):
                                if factor == symbol:
                                    # Get the other factor (should be 1/N)
                                    other_factor = arg.args[1 - i]

                                    # Handle Rational (like 1/64)
                                    if isinstance(other_factor, sp.Rational):
                                        divisor = other_factor.denominator
                                        if int(divisor) >= tid_upper_bound:
                                            floor_substitutions[expr] = 0

                                    # Handle Pow with -1 exponent (like 1/N)
                                    elif (
                                        isinstance(other_factor, sp.Pow)
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
    expression: sp.Expr, tid_upper_bound: int
) -> sp.Expr:
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


def _create_dimension_substitutions(dimension_values: List) -> Dict[sp.Symbol, sp.Expr]:
    """Create substitutions for dimension values (d0, d1, ...)."""
    substitutions = {}

    for i, value in enumerate(dimension_values):
        dimension_symbol = sp.Symbol(f"d{i}")
        if isinstance(value, int):
            substitutions[dimension_symbol] = sp.Integer(value)
        elif value in ["tid.x", "tid.y", "tid.z"]:
            substitutions[dimension_symbol] = sp.Symbol(
                value.replace(".", "_"), nonnegative=True
            )

    return substitutions


def _create_symbol_substitutions(symbol_values: List) -> Dict[sp.Symbol, sp.Expr]:
    """Create substitutions for symbol values (s0, s1, ...)."""
    substitutions = {}

    for i, value in enumerate(symbol_values):
        symbol_symbol = sp.Symbol(f"s{i}")
        if isinstance(value, int):
            substitutions[symbol_symbol] = sp.Integer(value)
        elif value in ["tid.x", "tid.y", "tid.z"]:
            substitutions[symbol_symbol] = sp.Symbol(
                value.replace(".", "_"), nonnegative=True
            )

    return substitutions


def simplify_expression(
    map_attr: AffineMapAttr,
    tid_upper_bound: int,
    dim_values: List = None,
    symbol_values: List = None,
) -> Optional[sp.Expr]:
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
        return sp.simplify(sympy_expr)

    except Exception:
        # If SymPy conversion fails, return None
        return None


def _emit_constant_expression(
    expression: sp.Expr, emitter, kernel_info, destination_register: str
) -> str:
    """Emit ASM for a constant expression."""
    from .instructions import VMovB32

    try:
        constant_value = int(expression)
        emitter.emit_instruction(VMovB32(2, constant_value))
        emitter.register_file.v_used.add(2)
        return destination_register
    except:
        # If evaluation fails, throw an error
        raise ValueError(
            f"Cannot evaluate constant expression: {expression}. Expected a constant integer value."
        )


def _emit_simple_thread_id_expression(emitter, kernel_info) -> str:
    """Emit ASM for a simple tid_x expression."""
    emitter.ensure_lane_id(kernel_info.subgroup_size)
    return "v0"  # lane ID register


def _emit_linear_thread_id_expression(
    expression: sp.Expr, emitter, kernel_info, destination_register: str
) -> str:
    """Emit ASM for linear expressions of the form a*tid_x + b."""
    from .instructions import VMovB32, VAddU32, VMulLoU32

    tid_x_symbol = sp.Symbol("tid_x", nonnegative=True)

    if not expression.is_polynomial(tid_x_symbol):
        return None

    coefficients = sp.Poly(expression, tid_x_symbol).coeffs()
    if len(coefficients) != 2:  # Not linear
        return None

    coefficient_a, coefficient_b = coefficients

    # Ensure lane ID is available
    emitter.ensure_lane_id(kernel_info.subgroup_size)

    if coefficient_a == 1 and coefficient_b == 0:
        # tid_x -> use lane ID directly
        return "v0"
    elif coefficient_a == 1:
        # tid_x + b -> add constant
        if coefficient_b != 0:
            emitter.emit_instruction(VMovB32(1, int(coefficient_b)))
            emitter.rf.v_used.add(1)
            emitter.emit_instruction(VAddU32(2, 0, 1))
            emitter.rf.v_used.add(2)
        return destination_register
    elif coefficient_a != 0:
        # a*tid_x + b -> multiply by a, add b
        if coefficient_a != 1:
            emitter.emit_instruction(VMovB32(1, int(coefficient_a)))
            emitter.rf.v_used.add(1)
            emitter.emit_instruction(VMulLoU32(2, 0, 1))
            emitter.rf.v_used.add(2)
        if coefficient_b != 0:
            emitter.emit_instruction(VMovB32(1, int(coefficient_b)))
            emitter.rf.v_used.add(1)
            emitter.emit_instruction(VAddU32(2, 2, 1))
        return destination_register

    return None


def _emit_thread_id_expression(
    expression: sp.Expr, emitter, kernel_info, destination_register: str
) -> str:
    """Emit ASM for expressions involving thread ID symbols."""
    tid_x_symbol = sp.Symbol("tid_x", nonnegative=True)

    # Handle simple case: just tid_x
    if expression == tid_x_symbol:
        return _emit_simple_thread_id_expression(emitter, kernel_info)

    # Handle linear expressions: a*tid_x + b
    linear_result = _emit_linear_thread_id_expression(
        expression, emitter, kernel_info, destination_register
    )
    if linear_result is not None:
        return linear_result

    # If we can't handle this expression, throw an error
    raise ValueError(
        f"Cannot emit ASM for thread ID expression: {expression}. Only simple tid_x and linear expressions of the form a*tid_x + b are supported."
    )


def emit_expression_asm(expr: sp.Expr, emitter, ki, dst_reg: str = "v2") -> str:
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


def _mlir_affine_to_sympy(expr) -> Optional[sp.Expr]:
    """Convert MLIR affine expression to SymPy expression by parsing the string representation."""
    try:
        expr_str = str(expr)

        # Handle simple cases first
        if expr_str.isdigit():
            return sp.Integer(int(expr_str))

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


def affine_map_simplifies_to_tid_x(
    map_attr: AffineMapAttr, tid_upper_bound: int
) -> bool:
    """Check if affine map simplifies to tid.x using SymPy for expression analysis."""
    simplified = simplify_expression(map_attr, tid_upper_bound)
    if simplified is None:
        return False

    # Check if the simplified expression is just tid_x
    tid_x = sp.Symbol("tid_x", nonnegative=True)
    return simplified == tid_x
