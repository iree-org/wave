# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from dataclasses import dataclass
from os import environ
from typing import Any, Callable, ClassVar, List, Optional, Type

import sympy
import torch.fx as fx

from .ir_utils import (
    is_float_type,
    is_integer_like_type,
)

from wave_lang.kernel.ops.wave_ops import get_custom
from wave_lang.kernel.lang import Memory
from wave_lang.kernel.lang.kernel_buffer import KernelBuffer
from wave_lang.kernel.compiler.utils import strides_from_symbolic_shape
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.support.logging import get_logger
from wave_lang.support.ir_imports import (
    AffineExpr,
    AffineMap,
    Attribute,
    DenseElementsAttr,
    F16Type,
    F32Type,
    F64Type,
    FlatSymbolRefAttr,
    FloatAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MemRefType,
    OpResult,
    Operation,
    ShapedType,
    StridedLayoutAttr,
    TypeAttr,
    UnitAttr,
    Value,
    VectorType,
    affine_d,
    arith_d,
    func_d,
    gpu_d,
    llvm_d,
    memref_d,
    vector_d,
)

from ..._support.indexing import IndexExpr, IndexingContext, xor
from ..._support.tracing import CapturedTrace
from ...compiler.base import NDEBUG, CodegenError
from ...compiler.builder import IRProxyValue, ScalarBuilder
from ...compiler.kernel_codegen import (
    BindingType,
    BindingDesc,
    BoundKernelSignature,
    create_argument_locations,
)
from ...lang.wave_types import IndexSymbol
from ..compile_options import WaveCompileOptions
from ..constraints import Constraint, HardwareConstraint, TilingConstraint
from ..utils.general_utils import get_hardware_constraint
from ..utils.symbol_utils import subs_idxc, is_literal

logger = get_logger("wave.ops_location_check")


def _get_upper_bound(expr: Any) -> Optional[Attribute]:
    res = subs_idxc(expr)
    if is_literal(res):
        return IntegerAttr.get(IndexType.get(), int(res))
    else:
        return None


@dataclass
class WaveEmitter:
    """Emits a warp function as a `func` with a signature derived from the gm."""

    root_sig: BoundKernelSignature
    trace: CapturedTrace
    constraints: list[Constraint]
    options: WaveCompileOptions
    grid: list[IndexExpr]
    kernel_name: str
    OP_HANDLERS: ClassVar[dict[str, Callable[["WaveEmitter", fx.Node], None]]] = {}
    _node_values: ClassVar[dict[fx.Node, List[IRProxyValue]]] = {}

    def __post_init__(self):
        self.dynamic_symbols = self.options.dynamic_symbols
        self.induction_vars: dict[IndexSymbol, Value] = {}
        self.dynamic_dims: dict[IndexSymbol, Value] = {}

    def emit_program_invariants(self):
        grid = self.grid

        self.workgroup_ids = [
            gpu_d.block_id(
                gpu_d.Dimension.x,
                upper_bound=_get_upper_bound(grid[0]),
            ),
            gpu_d.block_id(
                gpu_d.Dimension.y,
                upper_bound=_get_upper_bound(grid[1]),
            ),
            gpu_d.block_id(
                gpu_d.Dimension.z,
                upper_bound=_get_upper_bound(grid[2]),
            ),
        ]

        threads_per_block = self.hardware_constraint.threads_per_block
        self.thread_ids = [
            gpu_d.thread_id(
                gpu_d.Dimension.x, upper_bound=_get_upper_bound(threads_per_block[0])
            ),
            gpu_d.thread_id(
                gpu_d.Dimension.y, upper_bound=_get_upper_bound(threads_per_block[1])
            ),
            gpu_d.thread_id(
                gpu_d.Dimension.z, upper_bound=_get_upper_bound(threads_per_block[2])
            ),
        ]

    def emit_func(self) -> Operation:
        bindings = self.root_sig.sig.linear_bindings

        def abi_type(binding: BindingDesc):
            if binding.binding_type == BindingType.KERNEL_BUFFER:
                # Buffer passed to kernel as 0D memrefs to simplify ABI.
                element_type = IrType.parse(
                    binding.kernel_buffer_type.dtype.ir_type_asm()
                )
                return MemRefType.get([], element_type=element_type)

            # Scalars are passed as is.
            return binding.as_mlir_type()

        arg_types = [abi_type(b) for b in bindings]

        ftype = FunctionType.get(arg_types, [])
        func_op = func_d.FuncOp(self.kernel_name, ftype, visibility="private")

        locs = create_argument_locations(bindings)
        entry_block = func_op.add_entry_block(locs)

        # Map dynamic symbols to buffer argument indices and dimensions.
        for bind, arg in zip(bindings, entry_block.arguments):
            if bind.binding_type == BindingType.SYMBOL_VALUE:
                self.dynamic_dims[bind.symbol_type] = arg

        with InsertionPoint(entry_block), Location.name("wave-generated function"):
            self.emit_program_invariants()
            for bind, arg in zip(bindings, entry_block.arguments):
                node = bind.reference[1]
                if bind.binding_type != BindingType.KERNEL_BUFFER:
                    # Map scalar arguments to their corresponding node values.
                    self._node_values[node] = [arg]
                    continue

                dyn_val = MemRefType.get_dynamic_size()

                def get_static_dim(s: Optional[IndexExpr]) -> int:
                    if s is None:
                        return dyn_val

                    s = subs_idxc(s)
                    if is_literal(s):
                        return int(s)

                    return dyn_val

                # reinterpret_cast the 0D input buffers to the real shape/strides.
                element_type = IrType.parse(bind.kernel_buffer_type.dtype.ir_type_asm())
                symbolic_shape = bind.kernel_buffer_type.symbolic_shape
                physical_shape = symbolic_shape
                if layout := node.type.physical_layout:
                    physical_shape = layout.shape

                static_sizes = [get_static_dim(s) for s in physical_shape]

                idx_context = IndexingContext.current()
                strides = strides_from_symbolic_shape(
                    idx_context, physical_shape, allow_mixed_shapes=True
                )
                static_strides = [get_static_dim(s) for s in strides]
                layout = StridedLayoutAttr.get(offset=dyn_val, strides=static_strides)
                memref_type = MemRefType.get(static_sizes, element_type, layout=layout)

                offset = arith_d.constant(IndexType.get(), 0)
                dyn_sizes = [
                    gen_sympy_index(add_emitter_subs(self), s)
                    for s, p in zip(symbolic_shape, physical_shape)
                    if get_static_dim(p) == dyn_val
                ]
                dyn_strides = [
                    gen_sympy_index(add_emitter_subs(self), s)
                    for s in strides
                    if get_static_dim(s) == dyn_val
                ]
                res = memref_d.reinterpret_cast(
                    memref_type,
                    arg,
                    offsets=[offset],
                    sizes=dyn_sizes,
                    strides=dyn_strides,
                    static_offsets=[dyn_val],
                    static_sizes=static_sizes,
                    static_strides=static_strides,
                )
                self._node_values[node] = [res]
            func_d.return_([])

        return func_op

    def emit(self, graph: Optional[fx.Graph] = None) -> Operation:
        func = self.emit_func()
        with InsertionPoint.at_block_terminator(func.entry_block), Location.unknown():
            self._emit_graph(
                graph if graph is not None else self.trace.get_root_graph()
            )

        return func

    def _declare_runtime_func(
        self,
        name: str,
        arg_types: list[IrType],
        result_types: list[IrType],
        emit_c_interface: bool = False,
    ) -> tuple[Operation, FlatSymbolRefAttr]:
        """Helper to declare a runtime function and return both the func and its symbol."""
        ftype = FunctionType.get(arg_types, result_types)
        func_op = func_d.FuncOp(name, ftype, visibility="private")
        if emit_c_interface:
            func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        symbol = FlatSymbolRefAttr.get(func_op.sym_name.value)
        return func_op, symbol

    def emit_host_func(self, kernel_func: Operation) -> Operation:
        # TODO: kernel bindings order may not be the same as the kernel function
        # arguments order, so map kernel order to host function arguments order.
        binding_map = {}
        symbol_map = {}

        # SYMBOL_VALUE and INDEX_VALUE are determined by the kernel tensor args
        # shapes so they are not included in the host function arguments.
        host_bindings = [
            b
            for b in self.root_sig.sig.bindings
            if b.binding_type not in (BindingType.SYMBOL_VALUE, BindingType.INDEX_VALUE)
        ]
        for i, b in enumerate(host_bindings):
            binding_map[id(b)] = i
            if b.binding_type == BindingType.KERNEL_BUFFER:
                for j, symbol in enumerate(b.kernel_buffer_type.symbolic_shape):
                    if symbol in symbol_map:
                        continue
                    symbol_map[symbol] = (i, j)

        bindings = self.root_sig.sig.linear_bindings

        ptr = llvm_d.PointerType.get()

        def abi_type(binding: BindingDesc):
            if binding.binding_type == BindingType.KERNEL_BUFFER:
                # Buffer passed to kernel as 0D memrefs to simplify ABI.
                element_type = IrType.parse(
                    binding.kernel_buffer_type.dtype.ir_type_asm()
                )
                return MemRefType.get([], element_type=element_type)

            # Scalars are passed as is.
            return binding.as_mlir_type()

        arg_types = [abi_type(b) for b in bindings]

        ftype = FunctionType.get(arg_types, [])
        locs = [a.location for a in kernel_func.body.blocks[0].arguments]

        gpu_module = gpu_d.module("gpu_module")
        gpu_module.parent.operation.attributes["gpu.container_module"] = UnitAttr.get()
        module_block = gpu_module.bodyRegion.blocks.append()

        # TODO: propagate locations from the kernel
        with InsertionPoint(module_block), Location.name("wave-generated gpu module"):
            # TODO: GPUFuncOp doesn't seem to have a convenoent contructor yet
            kernel_func_wrapper = gpu_d.GPUFuncOp(
                TypeAttr.get(ftype), sym_name=self.kernel_name, kernel=True
            )

        new_kernel_entry_block = kernel_func_wrapper.body.blocks.append(
            *arg_types,
            arg_locs=locs,
        )

        # Inline the kernel function into the gpu module function body and erase the original function
        with (
            InsertionPoint(new_kernel_entry_block),
            Location.name("wave-generated kernel function"),
        ):
            # Move operations except terminator
            ops_to_move = list(kernel_func.entry_block)[:-1]
            for op in ops_to_move:
                op.detach_from_parent()
                new_kernel_entry_block.append(op)

            # Replace all uses of old arguments
            for old_arg, new_value in zip(
                kernel_func.entry_block.arguments, new_kernel_entry_block.arguments
            ):
                old_arg.replace_all_uses_with(new_value)

            gpu_d.return_([])
            kernel_func.erase()

        # Declare runtime functions
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        f64 = F64Type.get()

        buffer_type = MemRefType.get(
            [MemRefType.get_dynamic_size()], element_type=IntegerType.get_signless(8)
        )

        # Get buffer from PyObject*
        get_buffer_func, get_buffer_func_symbol = self._declare_runtime_func(
            "wave_get_buffer", [ptr], [buffer_type], emit_c_interface=True
        )

        # Get tensor dimension function from PyObject*
        get_dim_func, get_dim_func_symbol = self._declare_runtime_func(
            "wave_get_dim", [ptr, i32], [i64], emit_c_interface=True
        )

        # Scalar extraction functions from PyObject*
        get_int64_func, get_int64_func_symbol = self._declare_runtime_func(
            "wave_get_int64", [ptr], [i64], emit_c_interface=True
        )

        get_float64_func, get_float64_func_symbol = self._declare_runtime_func(
            "wave_get_float64", [ptr], [f64], emit_c_interface=True
        )

        # Declare host function
        # First argument is stream pointer
        # Rest are kernel arguments as PyObject*
        host_args_types = [ptr] + [ptr] * len(host_bindings)
        host_ftype = FunctionType.get(host_args_types, [])
        func_name = self.options.func_name
        func_op = func_d.FuncOp(func_name, host_ftype)

        locs = [Location.name("stream")] + create_argument_locations(host_bindings)
        entry_block = func_op.add_entry_block(locs)

        symbol_vals = {}

        subs = add_emitter_subs(self)
        threads_per_block = self.hardware_constraint.threads_per_block
        with InsertionPoint(entry_block), Location.name("wave-generated host function"):
            func_args = entry_block.arguments[1:]
            # Populate dynamic symbols from kernel function arguments
            for binding in bindings:
                if binding.binding_type != BindingType.SYMBOL_VALUE:
                    continue

                sym = binding.symbol_type
                arg_idx, dim_idx = symbol_map[sym]
                arg = func_args[arg_idx]
                dim = arith_d.constant(i32, dim_idx)
                value = func_d.call(
                    get_dim_func.type.results, get_dim_func_symbol, [arg, dim]
                )
                value = arith_d.index_cast(IndexType.get(), value)
                symbol_vals[sym] = value
                subs[sym] = value

            grid = [gen_sympy_index(subs, s) for s in self.grid]
            threads = [gen_sympy_index(subs, s) for s in threads_per_block]

            # Populate launch arguments
            launch_args = []
            for binding, dst_type in zip(bindings, arg_types):
                if binding.binding_type == BindingType.KERNEL_BUFFER:
                    arg = func_args[binding_map[id(binding)]]
                    # Extract buffer from PyObject
                    buffer = func_d.call(
                        get_buffer_func.type.results, get_buffer_func_symbol, [arg]
                    )
                    offset = arith_d.constant(IndexType.get(), 0)
                    buffer = memref_d.view(dst_type, buffer, offset, [])
                    launch_args.append(buffer)
                elif binding.binding_type == BindingType.SCALAR_VALUE:
                    arg = func_args[binding_map[id(binding)]]
                    # Extract scalar from PyObject
                    # Determine if it's float or int based on MLIR type
                    if isinstance(dst_type, (F16Type, F32Type, F64Type)):
                        # Float type - call wave_get_float64 and cast
                        scalar = func_d.call(
                            get_float64_func.type.results,
                            get_float64_func_symbol,
                            [arg],
                        )
                        if not isinstance(dst_type, F64Type):
                            scalar = arith_d.truncf(dst_type, scalar)
                    elif isinstance(dst_type, (IndexType, IntegerType)):
                        # Integer/Index type - call wave_get_int64 and cast
                        scalar = func_d.call(
                            get_int64_func.type.results, get_int64_func_symbol, [arg]
                        )
                        # Cast to target type if needed
                        if isinstance(dst_type, IndexType):
                            scalar = arith_d.index_cast(dst_type, scalar)
                        elif isinstance(dst_type, IntegerType) and dst_type.width != 64:
                            scalar = arith_d.trunci(dst_type, scalar)
                    else:
                        raise CodegenError(f"Unsupported scalar type: {dst_type}")

                    launch_args.append(scalar)

                elif binding.binding_type == BindingType.SYMBOL_VALUE:
                    sym = binding.symbol_type
                    value = symbol_vals[sym]
                    launch_args.append(value)
                else:
                    raise CodegenError(f"Unsupported binding type: {binding}")

            gpu_d.launch_func(
                async_dependencies=[],
                kernel=[gpu_module.sym_name.value, self.kernel_name],
                grid_size=grid,
                block_size=threads,
                kernel_operands=launch_args,
            )
            func_d.return_([])

        return func_op

    def _emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function" or node.op == "call_method":
                self._emit_function_call_node(node)
            if node.op == "output":
                return node.args

    def _emit_function_call_node(self, node: fx.Node):
        target_op = node.tkw_op_name
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")

        location = getattr(
            node, "location", None
        )  # type: Optional[FileLineColInfo | StackTraceInfo]

        try:
            if self.options.drop_debug_info_before_mlir:
                # Don't use any location context to avoid printing locations in MLIR
                handler(self, node)
            else:
                ir_location = location.to_mlir() if location else Location.unknown()
                with ir_location:
                    handler(self, node)
        except Exception as e:
            logger.error(f"Error handling {node}")
            raise e

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        assert NDEBUG or isinstance(node, fx.Node)
        values = self._node_values.get(node)
        if values is None:
            raise CodegenError(f"Node {node} has no IR Value")

        values = [v.ir_value if isinstance(v, IRProxyValue) else v for v in values]
        return values

    def bind_node_proxy(self, node: fx.Node, proxy: IRProxyValue):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        self._node_values[node] = [proxy]

    def bind_node_proxies(self, node: fx.Node, proxies: List[IRProxyValue]):
        assert NDEBUG or (
            isinstance(node, fx.Node)
            and all(isinstance(p, IRProxyValue) for p in proxies)
        )
        self._node_values[node] = proxies

    def get_induction_vars_and_syms(self) -> tuple[list[OpResult], list[IndexExpr]]:
        induction_var_syms = []
        induction_vars = []
        if self.induction_vars:
            for constraint in self.constraints:
                if isinstance(constraint, TilingConstraint):
                    if constraint.dim in self.induction_vars:
                        induction_var_syms.append(constraint.induction_var)
                        induction_vars.append(self.induction_vars[constraint.dim])

        return induction_vars, induction_var_syms

    @property
    def hardware_constraint(self) -> HardwareConstraint:
        return get_hardware_constraint(self.constraints)


def handle_op(op: Callable[..., Any] | list[Callable[..., Any]]):
    def decorator(
        f: Callable[[WaveEmitter, fx.Node], None],
    ) -> Callable[[WaveEmitter, fx.Node], None]:
        if isinstance(op, Callable):
            WaveEmitter.OP_HANDLERS[op.__name__] = f
        elif isinstance(op, list):
            for op_iter in op:
                WaveEmitter.OP_HANDLERS[op_iter.__name__] = f
        else:
            raise ValueError("handle_op only handle Callable or list of Callable")
        return f

    return decorator


def get_type_or_element_type(operand_type: IrType):
    assert isinstance(operand_type, IrType)
    if isinstance(operand_type, ShapedType):
        return operand_type.element_type
    else:
        return operand_type


def add_emitter_subs(
    emitter: WaveEmitter, dynamic_values: dict[IndexExpr, Value] = {}
) -> dict[IndexSymbol, Value]:
    induction_vars, induction_var_syms = emitter.get_induction_vars_and_syms()

    # TODO: factor this out
    # Add device dimension constants (always zero for now)
    device_zeros = [
        arith_d.constant(IndexType.get(), 0),  # DEVICE_DIM_0
        arith_d.constant(IndexType.get(), 0),  # DEVICE_DIM_1
        arith_d.constant(IndexType.get(), 0),  # DEVICE_DIM_2
    ]
    all_symbols = (
        emitter.thread_ids + emitter.workgroup_ids + device_zeros + induction_vars
    )
    dynamics = dict(
        zip(
            [
                THREAD_0,
                THREAD_1,
                THREAD_2,
                WORKGROUP_0,
                WORKGROUP_1,
                WORKGROUP_2,
                DEVICE_DIM_0,
                DEVICE_DIM_1,
                DEVICE_DIM_2,
            ]
            + induction_var_syms,
            all_symbols,
        )
    )
    dynamics.update(dynamic_values)
    dynamics.update(emitter.dynamic_dims)
    return dynamics


_emulate_ceildiv = bool(int(environ.get("WAVE_EMULATE_CEILDIV", 0)))
_use_affine_expr = bool(int(environ.get("WAVE_USE_AFFINE_EXPR", 1)))

_Rational = namedtuple("_Rational", ["numerator", "denominator"])
_ApplyExpr = namedtuple("_ApplyExpr", ["expr", "args"])


def gen_sympy_index(dynamics: dict[IndexSymbol, Value], expr: sympy.Expr) -> Value:
    use_affine_expr = _use_affine_expr
    stack: list[OpResult] = []

    def _get_ir_value(arg) -> Value:
        if isinstance(arg, _ApplyExpr):
            args = _broadcast(*arg.args)
            expr = arg.expr
            expr = AffineMap.get(dim_count=0, symbol_count=len(args), exprs=[expr])

            return affine_d.apply(expr, args)

        if not isinstance(arg, (Value, OpResult)):
            arg = arg.result

        return arg

    def _check_vec_scalar(a, b):
        if not isinstance(a, VectorType):
            return False

        if a.element_type == b:
            return True

        return (
            isinstance(b, VectorType)
            and b.shape == [1]
            and a.element_type == b.element_type
        )

    def _broadcast(*args) -> tuple[Value, ...]:
        assert len(args) > 0
        if len(args) == 1:
            return args

        res_args = [_get_ir_value(a) for a in args]
        res_type = res_args[0].type
        for arg in res_args[1:]:
            arg_type = arg.type
            if arg_type == res_type:
                continue

            if _check_vec_scalar(res_type, arg_type):
                # broadcast to res_type
                continue

            if _check_vec_scalar(arg_type, res_type):
                res_type = arg_type
                continue

            raise CodegenError(f"Cannot broadcast {res_type} and {arg.type}")

        for i, arg in enumerate(res_args):
            if arg.type == res_type:
                continue

            if isinstance(arg.type, VectorType):
                arg = vector_d.extract(arg, static_position=[0], dynamic_position=[])

            res_args[i] = vector_d.broadcast(res_type, arg)

        assert all(arg.type == res_type for arg in res_args)

        return tuple(res_args)

    def get_const_val(arg):
        if isinstance(arg, OpResult):
            arg = arg.owner.opview

        if isinstance(arg, arith_d.ConstantOp):
            value = arg.attributes["value"]
            if isinstance(value, IntegerAttr):
                return int(value)

        return None

    overflow_flags = arith_d.IntegerOverflowFlags.nsw | arith_d.IntegerOverflowFlags.nuw

    def muli(lhs, rhs):
        if get_const_val(lhs) == 1:
            return rhs

        if get_const_val(rhs) == 1:
            return lhs

        return arith_d.muli(lhs, rhs, overflow_flags=overflow_flags)

    def addi(lhs, rhs):
        if get_const_val(lhs) == 0:
            return rhs

        if get_const_val(rhs) == 0:
            return lhs

        if (rhs_val := get_const_val(rhs)) and rhs_val < 0:
            # If rhs_val is negative, since it is of index type and we have
            # overflow flags, we need to express it as a sub to get the
            # correct result.
            return arith_d.subi(
                lhs, _get_const(-rhs_val), overflow_flags=overflow_flags
            )

        return arith_d.addi(lhs, rhs, overflow_flags=overflow_flags)

    def op_expr(lhs, rhs, op):
        if isinstance(lhs, _ApplyExpr):
            lhs_args = lhs.args
            lhs_expr = lhs.expr
        else:
            lhs_args = [lhs]
            lhs_expr = AffineExpr.get_symbol(0)

        if isinstance(rhs, _ApplyExpr):
            rhs_args = rhs.args
            rhs_expr = rhs.expr
        else:
            rhs_args = [rhs]
            rhs_expr = AffineExpr.get_symbol(0)

        args = lhs_args + rhs_args
        expr = op(lhs_expr, rhs_expr.shift_symbols(len(rhs_args), len(lhs_args)))
        return _ApplyExpr(expr, args)

    def check_index_types(*args):
        return all(
            isinstance(a, _ApplyExpr) or isinstance(a.type, IndexType) for a in args
        )

    def add_expr(lhs, rhs):
        if not use_affine_expr or not check_index_types(lhs, rhs):
            return addi(*_broadcast(lhs, rhs))

        return op_expr(lhs, rhs, lambda a, b: a + b)

    def muli_expr(lhs, rhs):
        if not use_affine_expr or not check_index_types(lhs, rhs):
            return muli(*_broadcast(lhs, rhs))

        return op_expr(lhs, rhs, lambda a, b: a * b)

    def rem_expr(lhs, rhs):
        if not use_affine_expr or not check_index_types(lhs, rhs):
            return arith_d.remsi(*_broadcast(lhs, rhs))

        return op_expr(lhs, rhs, lambda a, b: a % b)

    def floordiv_expr(lhs, rhs):
        if not use_affine_expr or not check_index_types(lhs, rhs):
            return arith_d.divsi(*_broadcast(lhs, rhs))

        return op_expr(lhs, rhs, lambda a, b: AffineExpr.get_floor_div(a, b))

    def ceildiv_expr(lhs, rhs):
        if not use_affine_expr or not check_index_types(lhs, rhs):
            if _emulate_ceildiv:
                # ceildivui(x, y) = x == 0 ? 0 : ((x - 1) / y) + 1
                one = _get_const(1)
                zero = _get_const(0)
                lhs_minus_one = arith_d.subi(*_broadcast(lhs, one))
                div = arith_d.divui(*_broadcast(lhs_minus_one, rhs))
                result = arith_d.addi(*_broadcast(div, one))
                cmp = arith_d.cmpi(arith_d.CmpIPredicate.eq, *_broadcast(lhs, zero))
                zero, result = _broadcast(zero, result)
                return arith_d.select(cmp, zero, result)
            else:
                return arith_d.ceildivsi(*_broadcast(lhs, rhs))

        return op_expr(lhs, rhs, lambda a, b: AffineExpr.get_ceil_div(a, b))

    # `x + (a/b)` transformed into `(x*b + a) / b`
    def _add(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli_expr(lhs.denominator, rhs)
            numerator = add_expr(numerator, lhs.numerator)
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli_expr(lhs, rhs.denominator)
            numerator = add_expr(numerator, rhs.numerator)
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            lhs_numerator = muli_expr(lhs.numerator, rhs.denominator)
            rhs_numerator = muli_expr(rhs.numerator, lhs.denominator)
            numerator = add_expr(lhs_numerator, rhs_numerator)
            denominator = muli_expr(lhs.denominator, rhs.denominator)
            return _Rational(numerator, denominator)
        else:
            return add_expr(lhs, rhs)

    # `x * (a/b)` transformed into `(x * a) / b`
    def _mul(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli_expr(lhs.numerator, rhs)
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli_expr(lhs, rhs.numerator)
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            numerator = muli_expr(lhs.numerator, rhs.numerator)
            denominator = muli_expr(lhs.denominator, rhs.denominator)
            return _Rational(numerator, denominator)
        else:
            return muli_expr(lhs, rhs)

    def _rem(lhs, rhs):
        assert not isinstance(lhs, _Rational) and not isinstance(rhs, _Rational)

        return rem_expr(lhs, rhs)

    def _floor(value):
        if not isinstance(value, _Rational):
            return value

        return floordiv_expr(value.numerator, value.denominator)

    def _ceiling(value):
        if not isinstance(value, _Rational):
            return value

        return ceildiv_expr(value.numerator, value.denominator)

    def _group_rationals(stack, count):
        """Group rationals and non-rationals args into 2 contiguous sets.

        This allows to mul/add all non-rationals first, reducing total number of ops.
        """
        rationals = []
        non_rationals = []
        for _ in range(count):
            val = stack.pop()
            if isinstance(val, _Rational):
                rationals.append(val)
            else:
                non_rationals.append(val)

        return non_rationals + rationals

    def _apply(args, func):
        assert len(args) > 0
        value = args[0]
        for val in args[1:]:
            value = func(value, val)

        return value

    def _enforce_non_rational(val, term):
        if isinstance(val, _Rational):
            raise CodegenError(f"Rational is not supported yet in '{type(term)}'")

    def _remove_denominators(lhs, rhs):
        """
        Converts     (z/x) < y -> z < x*y
              or     z < (y/x) -> z*x < y
              or (a/b) < (c/d) -> a*d < b*c
        """
        if isinstance(lhs, _Rational) and not isinstance(rhs, _Rational):
            rhs = _mul(lhs.denominator, rhs)
            lhs = lhs.numerator
        if isinstance(rhs, _Rational) and not isinstance(lhs, _Rational):
            lhs = _mul(rhs.denominator, lhs)
            rhs = rhs.numerator
        if isinstance(lhs, _Rational) and isinstance(rhs, _Rational):
            rhs = _mul(lhs.denominator, rhs.numerator)
            lhs = _mul(rhs.denominator, lhs.numerator)
        return lhs, rhs

    def _get_const(val):
        if isinstance(val, int):
            return arith_d.constant(IndexType.get(), val)

        if isinstance(val, (tuple, list)):
            vec_type = VectorType.get([len(val)], IndexType.get())
            vals = [IntegerAttr.get(IndexType.get(), v) for v in val]
            return arith_d.constant(vec_type, DenseElementsAttr.get(vals, vec_type))

        raise CodegenError(f"Unsupported const val {val} : {type(val)}")

    idxc = IndexingContext.current()
    # Substitute in frozen vars to simplify expression.
    if not isinstance(expr, sympy.Expr):
        expr = sympy.sympify(expr)
    expr = expr.subs(idxc.subs)
    # Why affine, for now simply create indexing expressions.
    # This can easily be adapted to affine expressions later.
    select_stack = []
    for term in sympy.postorder_traversal(expr):
        match term:
            case sympy.Symbol():
                res = idxc.get_val(term)
                if res is not None:
                    stack.append(_get_const(res))
                elif term in dynamics.keys():
                    stack.append(dynamics[term])
                else:
                    raise CodegenError(f"Unknown symbol {term}")
            case sympy.Integer():
                stack.append(_get_const(int(term)))
            case sympy.Mul():
                args = _group_rationals(stack, len(term.args))
                stack.append(_apply(args, _mul))
            case sympy.Add():
                args = _group_rationals(stack, len(term.args))
                stack.append(_apply(args, _add))
            case sympy.Mod():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                stack.append(_rem(lhs, rhs))
            case sympy.floor():
                stack.append(_floor(stack.pop()))
            case sympy.ceiling():
                stack.append(_ceiling(stack.pop()))
            case sympy.Rational():
                numerator = _get_const(term.p)
                denominator = _get_const(term.q)
                stack.append(_Rational(numerator, denominator))
            case sympy.LessThan():
                rhs = stack.pop()
                lhs = stack.pop()
                if isinstance(rhs, _Rational) or isinstance(lhs, _Rational):
                    lhs, rhs = _remove_denominators(lhs, rhs)
                res = arith_d.cmpi(arith_d.CmpIPredicate.sle, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.StrictLessThan():
                rhs = stack.pop()
                lhs = stack.pop()
                if isinstance(rhs, _Rational) or isinstance(lhs, _Rational):
                    lhs, rhs = _remove_denominators(lhs, rhs)
                res = arith_d.cmpi(arith_d.CmpIPredicate.slt, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.StrictGreaterThan():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.cmpi(arith_d.CmpIPredicate.sgt, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.GreaterThan():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.cmpi(arith_d.CmpIPredicate.sge, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.Eq():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.cmpi(arith_d.CmpIPredicate.eq, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.Ne():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.cmpi(arith_d.CmpIPredicate.ne, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.And():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.andi(*_broadcast(lhs, rhs))
                for _ in range(len(term.args) - 2):
                    operand = stack.pop()
                    _enforce_non_rational(operand, term)
                    res = arith_d.andi(*_broadcast(res, operand))
                stack.append(res)
            case sympy.Or():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.ori(*_broadcast(lhs, rhs))
                for _ in range(len(term.args) - 2):
                    operand = stack.pop()
                    _enforce_non_rational(operand, term)
                    res = arith_d.ori(*_broadcast(res, operand))
                stack.append(res)
            case sympy.Max():
                count = len(term.args)
                res = stack.pop()
                _enforce_non_rational(res, term)
                res = _get_ir_value(res)
                elem_type = get_type_or_element_type(res.type)
                for _ in range(count - 1):
                    operand = stack.pop()
                    _enforce_non_rational(operand, term)
                    operand = _get_ir_value(operand)
                    if is_integer_like_type(elem_type):
                        res = arith_d.maxsi(*_broadcast(res, operand))
                    else:
                        res = arith_d.maximumf(*_broadcast(res, operand))

                stack.append(res)
            case sympy.Min():
                count = len(term.args)
                res = stack.pop()
                _enforce_non_rational(res, term)
                res = _get_ir_value(res)
                elem_type = get_type_or_element_type(res.type)
                for _ in range(count - 1):
                    operand = stack.pop()
                    _enforce_non_rational(operand, term)
                    operand = _get_ir_value(operand)
                    if is_integer_like_type(elem_type):
                        res = arith_d.minsi(*_broadcast(res, operand))
                    else:
                        res = arith_d.minimumf(*_broadcast(res, operand))

                stack.append(res)
            case sympy.logic.boolalg.BooleanFalse():
                res = arith_d.constant(IntegerType.get_signless(1), 0)
                stack.append(res)
            case sympy.logic.boolalg.BooleanTrue():
                res = arith_d.constant(IntegerType.get_signless(1), 1)
                stack.append(res)
            case sympy.Pow():
                _, power = term.args
                exponent = stack.pop()
                base = stack.pop()
                # Only support integer powers for now.
                if not isinstance(power, sympy.Integer):
                    raise CodegenError(f"Expected integer power, got {power}")
                if power == 0:
                    stack.append(_get_const(1))
                    continue
                for _ in range(sympy.Abs(power) - 1):
                    base = arith_d.muli(base, base)
                if power < 0:
                    stack.append(_Rational(_get_const(1), base))
                else:
                    stack.append(base)
            case xor():
                lhs = stack.pop()
                rhs = stack.pop()
                _enforce_non_rational(lhs, term)
                _enforce_non_rational(rhs, term)
                rhs = _get_ir_value(rhs)
                lhs = _get_ir_value(lhs)
                elem_type = get_type_or_element_type(rhs.type)
                res = arith_d.xori(*_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.UnevaluatedExpr():
                continue
            case sympy.functions.elementary.piecewise.ExprCondPair():
                cond = stack.pop()
                expr = stack.pop()
                select_stack.append(cond)
                select_stack.append(expr)
                continue
            case sympy.Piecewise():
                # Last cond must be True
                assert (
                    term.args and term.args[-1][1] == True
                ), f"Unsupported piecewise {term}"
                # Number of (condition, expression) pairs
                num_cases = len(term.args)

                # Pop all (condition, expression) pairs from select_stack
                # They are stored as: cond1, expr1, cond2, expr2, ..., condN, exprN
                cases = []
                for _ in range(num_cases):
                    expr = select_stack.pop()
                    cond = select_stack.pop()
                    cases.append((cond, expr))

                # Reverse to process in correct order (first case to last case)
                cases.reverse()

                # Build nested select operations
                # Start with the last expression (typically the default/else case)
                result = cases[-1][1]

                # Work backwards through earlier cases to build nested selects
                # Piecewise((expr1, cond1), (expr2, cond2), (expr3, True)) becomes:
                # select(cond1, expr1, select(cond2, expr2, expr3))
                for i in range(len(cases) - 2, -1, -1):
                    cond, expr = cases[i]
                    result = arith_d.select(cond, *_broadcast(expr, result))

                stack.append(result)
                assert not select_stack, f"select_stack is not empty: {select_stack}"
            case _:
                raise CodegenError(f"Can not handle {type(term)} : {term}")

    if len(stack) != 1 or isinstance(stack[0], _Rational):
        raise CodegenError(f"Expected single result, got {len(stack)}")

    return _get_ir_value(stack[0])


def get_constant_attr(value: Any, element_type: IrType) -> Attribute:
    if is_integer_like_type(element_type):
        return IntegerAttr.get(element_type, int(value))
    if is_float_type(element_type):
        return FloatAttr.get(element_type, float(value))
    raise CodegenError(f"Cannot create a constant attribute for type `{element_type}`")


def cast_py_literal(emitter: WaveEmitter, value) -> Any:
    """Treats the given value as a Python literal.

    An exception will be raised if it cannot be computed statically.
    """
    if isinstance(value, IndexExpr):
        simplified = IndexingContext.current().simplify_expr(value)
        try:
            return int(simplified)
        except TypeError as e:
            raise CodegenError(
                f"Literal value required but got symbolic value requiring "
                f"dynamic resolution: {simplified}"
            ) from e
    elif isinstance(value, tuple):
        return tuple(cast_py_literal(emitter, v) for v in value)
    elif isinstance(value, list):
        return [cast_py_literal(emitter, v) for v in value]
    elif isinstance(value, dict):
        return {
            cast_py_literal(emitter, k): cast_py_literal(emitter, v)
            for k, v in value.items()
        }
    elif isinstance(value, (int, float, str)):
        return value


def cast_py_value(
    emitter: WaveEmitter, value: Value, element_type: Optional[IrType] = None
) -> IRProxyValue:
    """
    Converts the given value to an IR Value.
    If the value is a fx.Node, the result of the fx.Node should corresspond to
    exactly one IR Value.
    If the value is a constant, a constant value will be built for it.
    """
    if isinstance(value, fx.Node):
        try:
            node_values = emitter.lookup_node_values(value)
            assert len(node_values) == 1, f"Expected exactly one value for node {value}"
            return (
                node_values[0]
                if isinstance(node_values[0], IRProxyValue)
                else IRProxyValue(node_values[0])
            )
        except KeyError:
            raise CodegenError(f"Producer node `{value}` has no IR Value")
    elif isinstance(value, IndexExpr):
        simplified = IndexingContext.current().simplify_expr(value)
        try:
            value = int(simplified)
        except TypeError as e:
            raise CodegenError(
                f"Dynamically resolved symbolic values not yet implemented. Got: "
                f"{simplified}"
            ) from e
    element_type = IndexType.get() if element_type is None else element_type
    return ScalarBuilder.constant(value, element_type)


def cast_vector(emitter: WaveEmitter, value, *, element_type: Optional[IrType] = None):
    proxy_value = cast_py_value(emitter, value)

    # Cast scalar types correctly first.
    if element_type and not ShapedType.isinstance(proxy_value.ir_value.type):
        # Implicit scalar type promotion.
        proxy_value = ScalarBuilder.to_dtype(proxy_value, element_type)

    value = proxy_value.ir_value

    # After scalar promotion, promote to vector.
    if VectorType.isinstance(value.type):
        # Already a vector. Coerce or return.
        if element_type is not None:
            value = ScalarBuilder.to_dtype(proxy_value, element_type).ir_value
        # No target element_type.
        return value
    else:
        # Scalar -> vector.
        element_type = value.type
        vector_type = VectorType.get([], element_type)
        return vector_d.broadcast(vector_type, value)


def cast_scalar(emitter: WaveEmitter, value):
    proxy_value = cast_py_value(emitter, value)
    value = proxy_value.ir_value

    # After scalar promotion, promote to vector.
    if VectorType.isinstance(value.type):
        # Vector -> scalar.
        return vector_d.extract(value, static_position=[0], dynamic_position=[])
    else:
        # Already a scalar. Coerce or return.
        # No target element_type.
        return value


def cast_py_value(
    emitter: WaveEmitter, value: Value, element_type: Optional[IrType] = None
) -> IRProxyValue:
    """
    Converts the given value to an IR Value.
    If the value is a fx.Node, the result of the fx.Node should corresspond to
    exactly one IR Value.
    If the value is a constant, a constant value will be built for it.
    """
    if isinstance(value, fx.Node):
        try:
            node_values = emitter.lookup_node_values(value)
            assert len(node_values) == 1, f"Expected exactly one value for node {value}"
            return (
                node_values[0]
                if isinstance(node_values[0], IRProxyValue)
                else IRProxyValue(node_values[0])
            )
        except KeyError:
            raise CodegenError(f"Producer node `{value}` has no IR Value")
    elif isinstance(value, IndexExpr):
        simplified = IndexingContext.current().simplify_expr(value)
        try:
            value = int(simplified)
        except TypeError as e:
            raise CodegenError(
                f"Dynamically resolved symbolic values not yet implemented. Got: "
                f"{simplified}"
            ) from e
    element_type = IndexType.get() if element_type is None else element_type
    return ScalarBuilder.constant(value, element_type)


def cast_py_lvalue(emitter: WaveEmitter, py_value: fx.Node) -> tuple[Value, fx.Node]:
    if isinstance(py_value, fx.Node):
        try:
            node_values = emitter.lookup_node_values(py_value)
            assert (
                len(node_values) == 1
            ), f"Expected exactly one value for node {py_value}"
            return node_values[0], py_value
        except KeyError:
            raise CodegenError(f"Producer node `{py_value}` has no IR Value")
    else:
        raise CodegenError(
            f"Required a traced node in the graph. Got: {py_value} (type {type(py_value)})"
        )


def cast_kernel_buffer(
    emitter: WaveEmitter, kb
) -> tuple[Value, MemRefType, Type[KernelBuffer]]:
    """Casts a Python value of type KernelBuffer, which lowers to a MemRefType'd value."""
    value, node = cast_py_lvalue(emitter, kb)
    ir_type = value.type
    py_type = node.type
    if py_type is None:
        try:
            py_type = get_custom(node).type
        except:
            raise CodegenError(f"Could not find type for node {node}")

    if not MemRefType.isinstance(ir_type):
        raise CodegenError(
            f"Expected a KernelBuffer (aka. `memref`) but got `{ir_type}`"
        )

    if not (issubclass(py_type, KernelBuffer) or issubclass(py_type, Memory)):
        raise CodegenError(
            f"Expected an lvalue of type KernelBuffer but got '{py_type}' for node {node}"
        )

    return value, MemRefType(ir_type), py_type
