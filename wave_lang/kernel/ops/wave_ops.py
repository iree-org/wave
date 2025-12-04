from __future__ import annotations

import math
import builtins
import copy
import operator
import sys
from abc import ABC
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    Type,
    TypeVar,
    final,
)

import numpy as np
import torch.fx as fx
from typing_extensions import Self
from itertools import combinations

from .._support.dtype import DataType, i1, i32
from .._support.indexing import IndexExpr, IndexSequence, IndexSymbol
from .._support.location import capture_location, CapturedLocation
from .._support.regions import RegionGraph
from ..lang.global_symbols import *
from ..lang.kernel_buffer import AddressSpace
from ..lang.wave_types import IndexMapping, Memory, Register
from .base import OpDispatcher
from ..wave.constraints import Constraint

if TYPE_CHECKING:
    from ..wave.scheduling.resources import Operation

T = TypeVar("T", bound=Type[Any])
AccT = TypeVar("AccT")
CustomOpT = TypeVar("CustomOpT", bound="CustomOp")
PlaceholderT = TypeVar("PlaceholderT", bound="Placeholder")

# List of keywords that are ignored when creating a custom op from an fx.Node.
# An example of this is tag, which is only used for tagging operators
# for their use in the custom wave schedule.
IGNORED_KEYWORDS = ["tag"]


def read_meets_hw_transpose_requirements(
    read: Read, constraints: list[Constraint], target: str
) -> bool:
    from ..wave.minimize_global_loads import is_transposed_read
    from ..wave.utils.general_utils import find_index_bounds

    # Check if target architecture supports amdgpu.transpose_load
    # Only gfx950 and newer architectures support this operation
    if not target.startswith("gfx95") and not target.startswith("gfx12"):
        return False

    if read.bounds is not None:
        return False

    if len(read.type.symbolic_shape) <= 1:
        return False

    if not is_transposed_read(read):
        return False

    bounds = find_index_bounds(
        constraints, read.index, read.vector_shapes, read.type.symbolic_shape
    )

    if bounds is not None:
        return False

    if read.has_identity_mapping():
        return False

    if len(list(read.index.keys())) != 2:
        return False

    bitwidth = read.type.dtype.bitwidth()
    if bitwidth != 8 and bitwidth != 16:
        return False

    bits = read.elements_per_thread * bitwidth
    if bits == 0 or bits % 64 != 0:
        return False

    if read.memory_type.address_space != SHARED_ADDRESS_SPACE:
        return False

    return not read.mapping_dynamic_vals


# Stubs to enable type checking of the custom ops:
# This is currently hand-written and should in future be generated from the custom ops


def allocate(
    shape: tuple[IndexExpr],
    distributed_shape: tuple[IndexExpr],
    dtype: DataType,
    address_space: IndexSymbol,
) -> "Memory": ...


def self_index(
    dim: IndexExpr,
    dtype: DataType,
    elements_per_thread: Optional[IndexExpr | int] = None,
) -> "Register": ...


def extract(
    register: "Register",
    offsets: tuple[IndexExpr],
) -> "Register": ...


def extract_slice(
    register: "Register",
    offsets: tuple[IndexExpr],
    sizes: tuple[IndexExpr],
    strides: tuple[IndexExpr],
) -> "Register": ...


def set_wave_prio(priority: int): ...


def shared_memory_barrier(wait_async_ops: bool = False): ...


def shared_memory_barrier_signal(barId: int = 0, tensor_wait: bool = False): ...


def shared_memory_barrier_wait(barId: int = 0): ...


def memory_counter_wait(load=None, store=None, ds=None, exp=None): ...


def workgroup_barrier(): ...


def read(
    memory: "Memory",
    elements_per_thread: Optional[IndexExpr | int] = None,
    mapping: Optional[IndexMapping] = None,
    mapping_dynamic_vals: "Register" | tuple["Register", ...] = (),
) -> "Register": ...


def conditional(
    condition: "Register" | IndexExpr,
    else_return: Optional[Sequence["Register"]] = None,
) -> Callable[[Callable[[], None]], None]: ...


def iterate(
    axis: IndexExpr, init_args: Sequence["Register"]
) -> Callable[[Callable[[AccT], AccT]], AccT]: ...


def register(
    shape: tuple[IndexExpr, ...], dtype: DataType, value: float
) -> "Register": ...


def scalar(value: float | int, dtype: DataType) -> "Register": ...


def mma(lhs: "Register", rhs: "Register", acc: "Register") -> "Register": ...


def scaled_mma(
    lhs: "Register",
    lhs_scale: "Register",
    rhs: "Register",
    rhs_scale: "Register",
    acc: "Register",
) -> "Register": ...


def write(
    register_: "Register",
    memory: "Memory",
    elements_per_thread: Optional[IndexExpr | int] = None,
    mapping: Optional[IndexMapping] = None,
    mapping_dynamic_vals: "Register" | tuple["Register", ...] = (),
): ...


def debug_log(
    register_: "Register",
    label: Optional[str],
    extra_iteration_dimensions: Optional[
        list[tuple["IndexSymbol", "IndexSymbol", int]]
    ] = None,
    mapping: Optional[IndexMapping] = None,
    mapping_dynamic_vals: "Register" | tuple["Register", ...] = (),
    printer: Optional[Callable[[str, Any], Any]] = None,
    handler: Optional[Callable[[dict[str, Any]], Any]] = None,
): ...


def bounds_check(
    index_exprs: dict[IndexExpr, IndexSequence],
    bounds: dict[IndexSymbol, IndexExpr],
    mapping: Optional[IndexMapping] = None,
    mapping_dynamic_vals: tuple["Register", ...] = (),
    mask_bounds: Optional[dict[IndexSymbol, IndexExpr]] = None,
): ...


def apply_expr(
    value: "Register" | Sequence["Register"], expr: Callable
) -> "Register": ...


def set_symbol(symbol: IndexExpr, value: "Register"): ...


def exp(src: "Register") -> "Register": ...


def exp2(src: "Register") -> "Register": ...


def sqrt(src: "Register") -> "Register": ...


def rsqrt(src: "Register") -> "Register": ...


def log2(src: "Register") -> "Register": ...


def log10(src: "Register") -> "Register": ...


def reciprocal(src: "Register") -> "Register": ...


def abs(src: "Register") -> "Register": ...


def softsign(
    src: "Register",
    logit_cap: float = 30.0,
    apply_scaling: bool = False,
    head_dim: int = None,
) -> "Register": ...


def tanh_approx(src: "Register") -> "Register": ...


def tanh(src: "Register") -> "Register": ...


def cos(src: "Register") -> "Register": ...


def round(src: "Register") -> "Register": ...


def roundeven(src: "Register") -> "Register": ...


def sin(src: "Register") -> "Register": ...


def sinh(src: "Register") -> "Register": ...


def maximum(lhs: "Register", rhs: "Register") -> "Register": ...


def minimum(lhs: "Register", rhs: "Register") -> "Register": ...


def atan2(lhs: "Register", rhs: "Register") -> "Register": ...


def powf(lhs: "Register", rhs: "Register") -> "Register": ...


def mod(lhs: "Register", rhs: "Register") -> "Register": ...


def remf(lhs: "Register", rhs: "Register") -> "Register": ...


def cbrt(src: "Register") -> "Register": ...


def atomic_min(
    lhs: "Register",
    rhs: "Memory",
    elements_per_thread: Optional[IndexExpr | int] = None,
    mapping: Optional[IndexMapping] = None,
    mapping_dynamic_vals: "Register" | tuple["Register", ...] = (),
) -> "Register": ...


def atomic_add(
    lhs: "Register",
    rhs: "Memory",
    elements_per_thread: Optional[IndexExpr | int] = None,
    mapping: Optional[IndexMapping] = None,
    mapping_dynamic_vals: "Register" | tuple["Register", ...] = (),
) -> "Register": ...


def broadcast(
    arg: "Register", target_shape: Optional[Sequence[IndexExpr | int]] = None
) -> "Register": ...


def sum(
    src: "Register",
    acc: Optional["Register"] = None,
    dim: Optional[IndexExpr | int] = None,
) -> "Register": ...


def cumsum(
    src: "Register",
    acc: Optional["Register"] = None,
    dim: Optional[IndexExpr | int] = None,
) -> "Register": ...


def max(
    src: "Register",
    acc: Optional["Register"] = None,
    dim: Optional[IndexExpr | int] = None,
) -> "Register": ...


def min(
    src: "Register",
    acc: Optional["Register"] = None,
    dim: Optional[IndexExpr | int] = None,
) -> "Register": ...


def topk(
    src: "Register",
    k_dim: IndexExpr,
    dim: IndexExpr,
) -> tuple["Register", "Register"]: ...


def shuffle(src: "Register", offset: int, width: int) -> "Register": ...


def gt(lhs: "Register", rhs: "Register") -> "Register": ...


def ge(lhs: "Register", rhs: "Register") -> "Register": ...


def lt(lhs: "Register", rhs: "Register") -> "Register": ...


def le(lhs: "Register", rhs: "Register") -> "Register": ...


def eq(lhs: "Register", rhs: "Register") -> "Register": ...


def ne(lhs: "Register", rhs: "Register") -> "Register": ...


def cast(src: "Register", dtype: DataType) -> "Register": ...


def bitcast(src: "Register", dtype: DataType) -> "Register": ...


def permute(src: "Register", target_shape: Sequence[IndexExpr]) -> "Register": ...


def reshape(
    inputs: Sequence["Register"], target_vector_shape: dict[IndexSymbol, int]
) -> "Register": ...


def select(
    cond: "Register", if_true: "Register", if_false: "Register"
) -> "Register": ...


def scatter_add(
    register_src: "Register",
    register_idx: "Register",
    dim: IndexExpr,
    memory: "Memory",
    mapping: IndexMapping,
    elements_per_thread: Optional[int] = 1,
) -> "Register": ...


def tensor_load_to_lds(
    src: Memory,
    dst: Memory,
    element_type: DataType,
    distributed_shape: list[IndexExpr],
    shared_tile_index: int,
    global_tile_index: dict[IndexSymbol, IndexSequence],
    bounds: dict[IndexSymbol, IndexExpr],
): ...


def gather_to_lds(
    src: Memory,
    dst: Memory,
    src_idx: dict[IndexSymbol, IndexSequence],
    dst_idx: dict[IndexSymbol, IndexSequence],
    dtype: DataType,
    elements_per_thread: Optional[IndexExpr | int] = None,
    src_mapping: Optional[IndexMapping] = None,
    dst_mapping: Optional[IndexMapping] = None,
    src_mapping_dynamic_vals: "Register" | tuple["Register", ...] = (),
    dst_mapping_dynamic_vals: "Register" | tuple["Register", ...] = (),
): ...


def define_op(op_name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        cls.tkw_op_name = op_name

        def new_function(*args: Any, **kwargs: dict[str, Any]):
            dispatcher = OpDispatcher.current()
            try:
                handler = getattr(dispatcher, f"handle_{op_name}")
            except AttributeError:
                raise AttributeError(
                    f"The current OpDispatcher ({dispatcher}) does not register a handler for {op_name}"
                )

            return handler(*args, **kwargs)

        new_function.__name__ = op_name
        current_module = sys.modules[cls.__module__]
        setattr(current_module, op_name, new_function)
        cls._tracing_function = new_function

        return cls

    return decorator


def define_py_op(py_op: Callable) -> Callable[[T], T]:
    """
    Register python internal operators as custom ops.
    This overloads python operator specific functions such as __add__ of
    fx.Proxy with a handler in order to control the tracing of the operator and
    map it to a dynamically created sublclass of UnaryPyOp or BinaryPyOp.
    """
    op_name = py_op.__name__

    def decorator(cls: T) -> T:
        # define new subclass of cls to represent this op
        @dataclass
        class NewSubclass(cls):
            pass

        NewSubclass.tkw_op_name = op_name
        NewSubclass.__name__ = f"{op_name.capitalize()}"
        NewSubclass.__module__ = cls.__module__
        current_module = sys.modules[cls.__module__]
        setattr(current_module, NewSubclass.__name__, NewSubclass)

        original_handler = None
        # Some py operator has trailing "_", which needs to be removed
        # before reformatting to torch.fx.Proxy formats.
        # i.e `and_` -> `and`, `or_` -> `or`.
        fx_op_name = op_name.replace("_", "")
        if hasattr(fx.Proxy, f"__{fx_op_name}__"):
            original_handler = getattr(fx.Proxy, f"__{fx_op_name}__")

        def new_function(*args: Any, **kwargs: dict[str, Any]):
            dispatcher = None
            try:
                dispatcher = OpDispatcher.current()
            except IndexError:
                handler = original_handler

            if dispatcher:
                try:
                    handler = getattr(dispatcher, f"handle_{op_name}")
                except AttributeError:
                    handler = original_handler

            return handler(*args, **kwargs)

        if original_handler:
            new_function.__name__ = op_name
            NewSubclass._tracing_function = new_function
            setattr(fx.Proxy, f"__{fx_op_name}__", new_function)

        # Return cls unchanged so we can reuse the decorator to register more ops
        return cls

    return decorator


def define_interface_op(op_name: str) -> Callable[[T], T]:
    """
    Generate new subclass for op handling, deriving from the base interface class.
    Generated subclass can be used for emitting the op from compiler/python side,
    by calling the generated subclass name.

    The generated subclass name would be pascal case of the TKW op name. For example:
    "tkw.op_name" -> "OpName"
    "tkw.exp2" -> "Exp2"
    """

    def decorator(cls: T) -> T:
        # define new subclass of cls to represent this op
        @dataclass
        class NewSubclass(cls):
            pass

        NewSubclass.tkw_op_name = op_name
        pascal_op_name = op_name.replace("_", " ").title().replace(" ", "")
        NewSubclass.__name__ = f"{pascal_op_name}"
        NewSubclass.__module__ = cls.__module__
        current_module = sys.modules[NewSubclass.__module__]
        setattr(current_module, NewSubclass.__name__, NewSubclass)
        if cls.__name__ == NewSubclass.__name__:
            raise ValueError(
                f'Subclass cannot have same name as base interface class{cls.__name__}. Did you mean to use"define_op" instead.'
            )

        def new_function(*args: Any, **kwargs: dict[str, Any]):
            dispatcher = OpDispatcher.current()
            try:
                handler = getattr(dispatcher, f"handle_{op_name}")
            except AttributeError:
                raise AttributeError(
                    f"The current OpDispatcher ({dispatcher}) does not register a handler for {op_name}"
                )

            return handler(*args, **kwargs)

        new_function.__name__ = op_name
        setattr(current_module, op_name, new_function)
        NewSubclass._tracing_function = new_function
        return cls

    return decorator


def get_custom(node: fx.Node) -> "CustomOp":
    """Get the corresponding CustomOp for a given fx.Node."""
    if not isinstance(node, fx.Node):
        raise ValueError(f"Expected an fx.Node but got {type(node)}")

    # If the node was created as a CustomOp it has a corresponding field
    if hasattr(node, "tkw_op"):
        return node.tkw_op.from_fx_node(node)
    if node.op == "placeholder":
        return Placeholder.from_fx_node(node)
    if node.op == "output":
        return Output.from_fx_node(node)
    return Unknown.from_fx_node(node)


def has_same_custom_type(lhs_type: Memory, rhs_type: Memory) -> bool:
    same_shape = lhs_type.symbolic_shape == rhs_type.symbolic_shape
    same_dtype = lhs_type.dtype == rhs_type.dtype
    return same_shape and same_dtype


@dataclass
class CustomOp(ABC):
    """
    Base class for all custom fx nodes.
    """

    graph: Optional[fx.Graph] = field(default=None, init=False)
    fx_node: Optional[fx.Node] = field(default=None, init=False)
    tkw_op_name: str = field(default="unknown", init=False)
    _tracing_function: Optional[Callable[..., Any]] = field(default=None, init=False)

    @property
    def location(self) -> Optional[CapturedLocation]:
        return getattr(self.fx_node, "location", None)

    @location.setter
    def location(self, value: Optional[CapturedLocation]):
        if value:
            setattr(self.fx_node, "location", value)

    @property
    def tag(self) -> Optional[str]:
        return getattr(self.fx_node, "tag", None)

    @classmethod
    def from_fx_node(cls: Type[CustomOpT], node: fx.Node) -> CustomOpT:
        instance = cls(*node.args)
        instance.fx_node = node
        instance.graph = node.graph
        if hasattr(node, "index"):
            instance.index = node.index
        return instance

    def __post_init__(self):
        # Subclasses do not inherit hash and eq from the superclass
        self.__class__.__hash__ = CustomOp.__hash__
        self.__class__.__eq__ = CustomOp.__eq__

    def __hash__(self):
        return hash(self.fx_node)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CustomOp):
            return False
        return self.fx_node == other.fx_node

    def __str__(self) -> str:
        return self.custom_string({})

    def custom_string(self, value_map: dict[str, str]) -> str:
        # print all variables of the node apart from graph and fx_node
        ignore_list = ["fx_node", "graph"]
        vars_list = [
            f"{key}={value}"
            for key, value in vars(self).items()
            if key not in ignore_list and value is not None
        ]
        if hasattr(self.fx_node, "index") and self.fx_node.index:
            vars_list.append(f"index={self.fx_node.index}")
        if hasattr(self.fx_node, "subgraph_name") and self.fx_node.subgraph_name:
            vars_list.append(f"subgraph_name={self.fx_node.subgraph_name}")
        vars_str = ", ".join(vars_list)
        return f"{self.tkw_op_name}({vars_str}) type({self.fx_node.type})"

    def add_to_graph(
        self,
        region_graph: RegionGraph,
        type: Any = None,
        loc: Optional[CapturedLocation] = None,
        tag: Optional[str] = None,
    ) -> fx.Node:
        # Exclude tag from args since it's not part of the dataclass constructor
        arg_list = tuple(
            [value for key, value in vars(self).items() if key not in IGNORED_KEYWORDS]
        )
        self.graph = region_graph
        self.fx_node = region_graph.create_node(
            "call_function",
            target=self._tracing_function,
            args=arg_list,
            kwargs={},
        )
        self.fx_node.tkw_op = self.__class__
        self.fx_node.tkw_op_name = self.tkw_op_name
        self.fx_node.index = None
        if loc is not None:
            self.fx_node.location = loc
        if type is None:
            get_custom(self.fx_node).infer_type()
        else:
            self.fx_node.type = type
        if tag is not None:
            self.fx_node.tag = tag
        return self.fx_node

    def _add_proxy_to_graph(self, region_graph: RegionGraph):
        # Exclude tag from args since it's not part of the dataclass constructor
        arg_list = tuple(
            [value for key, value in vars(self).items() if key not in IGNORED_KEYWORDS]
        )
        self.graph = region_graph
        self.fx_node = region_graph.create_proxy(
            "call_function",
            target=self._tracing_function,
            args=arg_list,
            kwargs={},
        )

    def update_arg(self, idx_or_name: int | str | fx.Node, value: CustomOp | fx.Node):
        """
        Update the value of an argument in the node while keeping the
        underlying fx.Node consistent.
        """
        inherited_field_count = len(CustomOp.__dataclass_fields__)
        field_names = [field.name for field in fields(self)[inherited_field_count:]]
        if isinstance(idx_or_name, str):
            if idx_or_name not in field_names:
                raise ValueError(f"Field {idx_or_name} not found")
            idx = field_names.index(idx_or_name)
        elif isinstance(idx_or_name, fx.Node):
            idx = self.fx_node.args.index(idx_or_name)
        else:
            idx = idx_or_name
        if isinstance(value, CustomOp):
            value = value.fx_node
        # Skip the fields defined by the abstract base class
        if 0 <= idx < len(field_names):
            field_name = field_names[idx]
            # Set the new value for the field
            setattr(self, field_name, value)
            self.fx_node.update_arg(idx, value)
        else:
            raise IndexError("Index out of range")

    def copy_core_attributes(self, new_node: fx.Node):
        """
        Copy core attributes from the current node to the new node.
        """
        core_attributes = [
            "index",
            "vector_shapes",
            "reduction_dim",
            "iter_idx",
            "location",
            "expanded_dims",
            "scheduling_parameters",
        ]
        for attr_name in core_attributes:
            if hasattr(self.fx_node, attr_name):
                attr = getattr(self.fx_node, attr_name)
                if attr_name == "index":
                    attr = copy.deepcopy(attr)
                setattr(new_node, attr_name, attr)
        for key, value in self.fx_node.meta.items():
            new_node.meta[key] = value

    def copy(
        self,
        new_name: Optional[str] = None,
        new_graph: Optional[fx.Graph] = None,
        arg_transform: Optional[Callable[[Any], Any]] = lambda x: x,
        anchor: Optional[fx.Node] = None,
    ) -> Self:
        """Returns a duplicate of this node."""
        graph = new_graph
        if new_graph is None:
            graph = self.graph
            if anchor is None:
                anchor = self.fx_node
            graph.inserting_after(anchor)
        new_node = graph.node_copy(self.fx_node, arg_transform=arg_transform)
        new_node.tkw_op = self
        new_node.tkw_op_name = self.tkw_op_name
        new_node.tag = self.tag
        self.copy_core_attributes(new_node)
        if new_name:
            new_node.name = new_name
        return get_custom(new_node)

    def replacement_location_propagate(
        self, new_node: CustomOp | fx.Node | list[fx.Node]
    ):
        """Set the new_node location if it doesn't have one."""
        if isinstance(new_node, fx.Node):
            get_custom(new_node).location = self.location
        if isinstance(new_node, list):
            for node in new_node:
                custom = get_custom(node) if isinstance(node, fx.Node) else node
                custom.location = self.location
        else:
            new_node.location = self.location

    def replace_all_uses_with(self, new_node: CustomOp | fx.Node):
        """Replace all uses of the current node with the new node."""
        if isinstance(new_node, CustomOp):
            new_node = new_node.fx_node
        self.replacement_location_propagate(new_node)
        self.fx_node.replace_all_uses_with(new_node)

    def replace_all_uses_with_except(
        self, new_node: CustomOp | fx.Node, except_nodes: list[CustomOp]
    ):
        """Replace all uses of the current node with the new node except for the nodes in except_nodes."""
        if isinstance(new_node, fx.Node):
            new_node = get_custom(new_node)

        self.replacement_location_propagate(new_node)
        for user in self.users:
            if user in except_nodes:
                continue
            indices = user.get_node_arg_index(self)
            if not isinstance(indices, Sequence):
                indices = [indices]
            for idx in indices:
                if isinstance(user.node_args[idx], Sequence):
                    sub_idx = user.node_args[idx].index(self)
                    new_nodes = [
                        (
                            user.node_args[idx][x].fx_node
                            if x != sub_idx
                            else new_node.fx_node
                        )
                        for x in range(len(user.node_args[idx]))
                    ]
                    user.update_arg(idx, new_nodes)
                else:
                    user.update_arg(idx, new_node.fx_node)

    def erase(self):
        """Erase the current node from the graph where it exists."""
        assert (
            not self.fx_node.users
        ), f"Attempting to erase {self.fx_node} which has {self.fx_node.users} users!"
        self.graph.erase_node(self.fx_node)

    @classmethod
    def handle(cls, graph: RegionGraph, *args, **kwargs) -> fx.Node:
        # Extract tag from kwargs if present
        tag = kwargs.pop("tag", None)
        node = cls(*args, **kwargs)
        node._add_proxy_to_graph(graph)
        node.fx_node.node.tkw_op = cls
        node.fx_node.node.tkw_op_name = cls.tkw_op_name
        node.fx_node.node.location = capture_location(graph.location_capture_config)
        if tag is not None:
            node.fx_node.node.tag = tag
        return node.fx_node

    @property
    def name(self) -> str:
        if hasattr(self, "_name"):
            return self._name
        return self.fx_node.name

    @property
    def node_args(self) -> dict[int, Any]:
        """Returns the args to this custom op using subclasses of CustomOp if possible."""

        def propagate(n):
            custom = get_custom(n)
            if isinstance(custom, Placeholder):
                if prev := custom.get_captured_fx_node():
                    return propagate(prev)

            return custom

        custom_args = {}
        for i, arg in enumerate(self.fx_node.args):
            if isinstance(arg, fx.Node):
                custom_args[i] = propagate(arg)
            if isinstance(arg, Sequence) and all(isinstance(x, fx.Node) for x in arg):
                custom_args[i] = [propagate(x) for x in arg]
        return custom_args

    def get_node_arg_index(self, arg: CustomOp) -> Optional[CustomOp | list[CustomOp]]:
        keys = []
        for key, value in self.node_args.items():
            if isinstance(value, Sequence):
                if arg in value:
                    keys.append(key)
            elif value == arg:
                keys.append(key)
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        return keys

    @property
    def users(self) -> list[Any]:
        """Returns the users of this custom op using subclasses of CustomOp if possible."""
        return [get_custom(user) for user in self.fx_node.users]

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return []

    @property
    def index(self) -> Optional[dict[IndexSymbol, IndexSequence]]:
        if hasattr(self.fx_node, "index"):
            return self.fx_node.index
        return None

    @index.setter
    def index(self, value: Any):
        """
        Updates the index of the node based on a per-dimension index sequence.
        """
        if value is None:
            return
        if isinstance(value, dict):
            assert all(
                isinstance(v, IndexSequence) for v in value.values()
            ), f"Index must be a dict with values of type IndexSequence but got {value}"
            self.fx_node.index = {}
            for dim, key in value.items():
                self.fx_node.index[dim] = key
        elif isinstance(value, list):
            self.fx_node.index = value
        else:
            raise ValueError("Index must be a dict")

    @property
    def rrt(self):
        if hasattr(self.fx_node, "rrt"):
            return self.fx_node.rrt

    @rrt.setter
    def rrt(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("RRT must be a numpy array")
        self.fx_node.rrt = value

    @property
    def scheduling_parameters(self):
        if hasattr(self.fx_node, "scheduling_parameters"):
            return self.fx_node.scheduling_parameters

    @scheduling_parameters.setter
    def scheduling_parameters(self, value: Any):
        if not isinstance(value, dict):
            raise ValueError("Scheduling parameters must be a dict")
        self.fx_node.scheduling_parameters = value

    @property
    def expanded_dims(self) -> dict[IndexSymbol, int]:
        """
        During expansion each node is expanded along its indexing dimensions.
        The expanded_dims property stores the dimensions along which the node
        has been expanded as well as the scaling along that dimension.

        For example, a node with indexing dimensions [M, N] with
        dimensional scaling {M: 2, N: 2}, will be expanded to 4 nodes,
        with each expanded node mapping to the following expanded_dims
        {M: 0, N: 0}, {M: 0, N: 1}, {M: 1, N: 0}, {M: 1, N: 1}.
        """
        if hasattr(self.fx_node, "expanded_dims"):
            return self.fx_node.expanded_dims
        return None

    @expanded_dims.setter
    def expanded_dims(self, value: dict[IndexSymbol, int]):
        if not isinstance(value, dict):
            raise ValueError("Expanded dims must be a dict")
        self.fx_node.expanded_dims = value

    @property
    def vector_shapes(self) -> dict[IndexSymbol, int]:
        if hasattr(self.fx_node, "vector_shapes"):
            return self.fx_node.vector_shapes
        return None

    @vector_shapes.setter
    def vector_shapes(self, value: dict[IndexSymbol, int]):
        self.fx_node.vector_shapes = value

    @property
    def type(self) -> Any:
        if hasattr(self.fx_node, "type"):
            return self.fx_node.type
        return None

    @type.setter
    def type(self, value: Any):
        self.fx_node.type = value

    @property
    def has_side_effects(self) -> bool:
        return False

    @property
    def pre_expansion_id(self) -> int:
        if hasattr(self.fx_node, "pre_expansion_id"):
            return self.fx_node.pre_expansion_id
        return None

    @pre_expansion_id.setter
    def pre_expansion_id(self, value: int):
        self.fx_node.pre_expansion_id = value

    def infer_type(self, *args):
        """
        Infer the type of this operator using the types
        of its arguments.
        """
        pass

    def transform_index_backwards(
        self, index: dict[IndexSymbol, IndexSequence], arg: fx.Node
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Transform the index of the node when propagating index backwards, i.e.
        from node to its arguments.
        """
        return index

    def transform_index(
        self, index: dict[IndexSymbol, IndexSequence]
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Transform the index of the node based on the provided mapping.
        """
        return index


@dataclass
class BinaryOpBase(CustomOp, ABC):
    """
    Represents an elementwise binary python operator.

    DTYPE requirement: lhs and rhs needs to have the same dtpye.
    Shape requirement: lhs and rhs either have same shape or
                       their shape must be broadcastable to
                       one another.
    """

    lhs: Any
    rhs: Any

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        combined_dims = []
        if isinstance(self.lhs, fx.Node):
            combined_dims += get_custom(self.lhs).indexing_dims
        if isinstance(self.rhs, fx.Node):
            combined_dims += get_custom(self.rhs).indexing_dims

        unique_dims = list(dict.fromkeys(combined_dims))
        return unique_dims

    @property
    def py_operator(self) -> str:
        return self.tkw_op_name

    def infer_shape(self) -> Any:
        lhs_type = get_custom(self.lhs).type
        rhs_type = get_custom(self.rhs).type
        if isinstance(lhs_type, DataType) and isinstance(rhs_type, DataType):
            has_same_type = True
        else:
            has_same_type = has_same_custom_type(lhs_type, rhs_type)
        if has_same_type:
            return lhs_type.symbolic_shape

        lhs_dim_set = set(lhs_type.symbolic_shape)
        rhs_dim_set = set(rhs_type.symbolic_shape)
        if lhs_dim_set and rhs_dim_set and lhs_dim_set.isdisjoint(rhs_dim_set):
            raise ValueError(
                "BinaryPyOp requires lhs and rhs shape to be at least broadcastable."
                f" got {lhs_type.symbolic_shape} vs {rhs_type.symbolic_shape}"
            )

        # TODO: this logic looks suspicious. Specifically, there's no check that
        # rhs_dim_set subsumes lhs_dim_set, they may partially overlap.
        broadcasted_type = lhs_type if lhs_dim_set > rhs_dim_set else rhs_type
        return broadcasted_type.symbolic_shape


@define_py_op(operator.add)
@define_py_op(operator.sub)
@define_py_op(operator.mul)
@define_py_op(operator.and_)
@define_py_op(operator.or_)
@define_py_op(operator.truediv)
@define_py_op(operator.mod)
@define_interface_op("maximum")
@define_interface_op("minimum")
@define_interface_op("atan2")
@define_interface_op("powf")
@define_interface_op("remf")
@dataclass
class BinaryPyOp(BinaryOpBase, ABC):
    def infer_type(self, *args):
        if not self.infer_shape():
            # In scalar codegen, we do not have
            # the concept of a shape (yet).
            self.type = self.lhs.type
        else:
            self.type = Register[(*self.infer_shape(), get_custom(self.lhs).type.dtype)]


@define_py_op(operator.eq)
@define_py_op(operator.gt)
@define_py_op(operator.ge)
@define_py_op(operator.lt)
@define_py_op(operator.le)
@define_py_op(operator.ne)
@define_interface_op("eq")
@define_interface_op("gt")
@define_interface_op("ge")
@define_interface_op("lt")
@define_interface_op("le")
@define_interface_op("ne")
@dataclass
class ComparisonPyOp(BinaryOpBase, ABC):
    def infer_type(self, *args):
        # If lhs & rhs is scalar then shape is empty, then type is i1.
        # Else, output i1 with shape of lhs/rhs.
        shape = self.infer_shape()
        self.type = Register[(*shape, i1)] if shape else i1


@define_interface_op("abs")
@define_interface_op("exp")
@define_interface_op("exp2")
@define_interface_op("sqrt")
@define_interface_op("rsqrt")
@define_interface_op("log2")
@define_interface_op("log10")
@define_interface_op("reciprocal")
@define_interface_op("round")
@define_interface_op("roundeven")
@define_interface_op("sin")
@define_interface_op("sinh")
@define_interface_op("tanh")
@define_interface_op("tanh_approx")
@define_interface_op("cos")
@define_interface_op("cbrt")
@define_py_op(operator.neg)
@define_py_op(operator.invert)
@dataclass
class UnaryPyOp(CustomOp, ABC):
    """
    Represents a unary python operator.
    """

    arg: fx.Node

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    @property
    def py_operator(self) -> str:
        return self.tkw_op_name

    def infer_type(self, *args):
        src_type = get_custom(self.arg).type
        self.type = src_type


@define_interface_op("softsign")
@dataclass
class SoftsignOp(CustomOp, ABC):
    arg: fx.Node
    logit_cap: float = 30.0
    apply_scaling: bool = False
    head_dim: int = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    def infer_type(self, *args):
        src_type = get_custom(self.arg).type
        self.type = src_type


@define_op("select")
@dataclass
class SelectOp(CustomOp):
    cond: fx.Node
    if_true: fx.Node
    if_false: fx.Node

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        combined_dims = []
        combined_dims += get_custom(self.cond).indexing_dims
        combined_dims += get_custom(self.if_true).indexing_dims
        combined_dims += get_custom(self.if_false).indexing_dims
        return list(dict.fromkeys(combined_dims))

    def infer_broadcast_shape(
        self, cond_type: Register, t_type: Register, f_type: Register
    ):
        c_shape = getattr(cond_type, "symbolic_shape", ())
        t_shape = getattr(t_type, "symbolic_shape", ())
        f_shape = getattr(f_type, "symbolic_shape", ())

        # Check if shapes are broadcastable - they should not be disjoint
        non_empty_shapes = [s for s in [c_shape, t_shape, f_shape] if s]
        if any(
            set(i_shape).isdisjoint(set(j_shape))
            for i_shape, j_shape in combinations(non_empty_shapes, 2)
        ):
            raise ValueError(
                f"SelectOp requires operand shapes to be broadcastable. "
                f"Got condition: {c_shape}, if_true: {t_shape}, if_false: {f_shape}"
            )

        return builtins.max(non_empty_shapes, key=len, default=())

    def infer_type(self, *args):
        cond_type = get_custom(self.cond).type
        if_true_type = get_custom(self.if_true).type
        if_false_type = get_custom(self.if_false).type

        if cond_type.dtype != i1:
            raise ValueError("SelectOp expects condition type to be i1.")

        if if_true_type.dtype != if_false_type.dtype:
            raise ValueError("SelectOp expects lhs and rhs dtype to match.")

        broadcasted_shape = self.infer_broadcast_shape(
            cond_type, if_true_type, if_false_type
        )

        result_type = Register[broadcasted_shape, if_true_type.dtype]

        self.type = result_type


@final
@dataclass
class Unknown(CustomOp):
    """
    Represents an fx.Node that has no corresponding CustomNode class.
    """

    args: Sequence[Any]
    kwargs: dict[Any, Any]

    @classmethod
    def from_fx_node(cls, node: fx.Node) -> "Unknown":
        instance = cls(node.args, node.kwargs)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def custom_string(self, value_map: dict[str, str]) -> str:
        # print all variables of the node apart from graph and op
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][:-2]
        vars_str = ", ".join(vars_list)
        return f"unknown: {self.fx_node.name}({vars_str}) type({self.fx_node.type})"


@dataclass
class Output(CustomOp):
    """
    Represents an output node in the graph, representing the return value of a
    traced function.
    """

    return_vals: Sequence[Any]
    tkw_op_name: str = field(default="output", init=False)

    @classmethod
    def from_fx_node(cls: Type[CustomOpT], node: fx.Node) -> CustomOpT:
        instance = cls(node.args)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def add_to_graph(
        self, region_graph: RegionGraph, loc: Optional[CapturedLocation] = None
    ) -> fx.Node:
        self.graph = region_graph
        self.fx_node = region_graph.create_node(
            "output",
            target="output",
            args=tuple([self.return_vals]),
            kwargs={},
        )
        self.fx_node.tkw_op = self.__class__
        self.fx_node.tkw_op_name = self.tkw_op_name
        if loc is not None:
            self.fx_node.location = loc
        return self.fx_node

    @property
    def has_side_effects(self) -> bool:
        return True


@dataclass
class Placeholder(CustomOp):
    """
    Represents a placeholder node in the graph, i.e. an input to a function.
    """

    _name: str
    _type: Optional[Type[DataType] | Type[Memory]] = None
    tkw_op_name: str = field(default="placeholder", init=False)

    @classmethod
    def from_fx_node(cls: Type[PlaceholderT], node: fx.Node) -> PlaceholderT:
        instance = cls(node.name, node.type)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def add_to_graph(
        self, region_graph: RegionGraph, loc: Optional[CapturedLocation] = None
    ) -> fx.Node:
        self.graph = region_graph
        self.fx_node = region_graph.create_node("placeholder", target=self._name)
        self.fx_node.tkw_op = self.__class__
        self.fx_node.tkw_op_name = self.tkw_op_name
        if loc is not None:
            self.fx_node.location = loc
        return self.fx_node

    def custom_string(self, value_map: dict[str, str]) -> str:
        # print all variables of the node apart from graph and op
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][:-2]
        vars_str = ", ".join(vars_list)
        return f"{self.tkw_op_name}({vars_str}) type({self.fx_node.type})"

    def erase(self):
        """Erase the current node from the graph where it exists."""

        super().erase()
        if not hasattr(self.graph, "parent_op"):
            return

        parent = self.graph.parent_op
        custom = get_custom(parent)
        if not isinstance(custom, NestedRegionOp):
            return

        # Cleanup dead captures
        subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
        live_captures = []
        for var in custom.implicit_captures:
            if custom.get_captured_fx_node(subgraph, var):
                live_captures.append(var)

        custom.update_arg("implicit_captures", live_captures)

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        if not hasattr(self._type, "symbolic_shape"):
            return []
        return list(self._type.symbolic_shape) if self._type else []

    def get_captured_fx_node(self) -> Optional[fx.Node]:
        return self.fx_node.meta.get("lifted", None)

    def infer_type(self, *args):
        self.fx_node.type = self._type

    @property
    def index(self) -> list[dict[IndexSymbol, IndexSequence]]:
        var = self.get_captured_fx_node()
        if var is not None:
            return get_custom(var).index

        if hasattr(self.fx_node, "index"):
            return self.fx_node.index

        return None

    @index.setter
    def index(self, value: Any):
        var = self.get_captured_fx_node()
        if var is None:
            CustomOp.index.fset(self, value)
            return

        get_custom(var).index = value


@dataclass
class IterArg(Placeholder):
    """
    Represents a specific placeholder node in the graph that is an iter arg of
    a reduction node. IterArgs can be of type Register or Memory with
    a Shared memory address space.
    """

    def parent_op(self):
        return get_custom(self.graph.parent_op)

    @property
    def iter_idx(self):
        if hasattr(self.fx_node, "iter_idx"):
            return self.fx_node.iter_idx
        return None

    @iter_idx.setter
    def iter_idx(self, value):
        self.fx_node.iter_idx = value

    @property
    def distributed_shape(self):
        init_arg = self.parent_op().init_args[self.iter_idx]
        allocate = get_custom(init_arg)
        assert isinstance(allocate, Allocate)
        return allocate.distributed_shape

    def infer_type(self, *args):
        parent_op = self.parent_op()
        init_args = parent_op.init_args
        self.type = init_args[self.iter_idx].type


# Ops modeling TKW operations in the kernel language


@define_op("allocate")
@dataclass
class Allocate(CustomOp):
    """
    Represents an allocation in an address space (such as shared memory).
    """

    shape: tuple[IndexExpr]
    distributed_shape: tuple[IndexExpr]
    dtype: DataType
    address_space: AddressSpace = SHARED_ADDRESS_SPACE
    padding: int = 0
    parent: Optional[fx.Node] = None
    offset: Optional[IndexExpr] = None
    tail_padding: int = 0  # Padding after the array end

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        from ..wave.utils.general_utils import infer_dim

        shape = list(self.shape)
        dims = [infer_dim(expr) for expr in shape]
        return dims

    @property
    def type(self) -> "Memory":
        return Memory[(*self.shape, self.address_space, self.dtype)]

    @property
    def allocation_size(self) -> IndexExpr:
        """
        Returns the full size of the allocation in bytes including all padding.
        """
        return (
            (math.prod(self.distributed_shape) + self.tail_padding)
            * self.dtype.bitwidth()
        ) // 8

    @property
    def unpadded_dims(self) -> dict[IndexSymbol, IndexExpr]:
        from ..wave.utils.general_utils import is_scaled_dim, infer_dim

        unpadded_dim = {}
        last_sym_type = self.type.symbolic_shape[-1]
        last_sym_type = (
            infer_dim(self.type.symbolic_shape[-1])
            if is_scaled_dim(last_sym_type)
            else last_sym_type
        )
        unpadded_dim[last_sym_type] = self.distributed_shape[-1] - self.padding
        for idx, sym_type in enumerate(self.type.symbolic_shape[:-1]):
            sym_type = infer_dim(sym_type) if is_scaled_dim(sym_type) else sym_type
            unpadded_dim[sym_type] = self.distributed_shape[idx]
        return unpadded_dim

    @property
    def unpadded_shape(self) -> tuple[IndexExpr]:
        unpadded_dims = self.unpadded_dims
        return tuple(unpadded_dims[s] for s in self.shape)

    def infer_type(self, *args):
        type_expr = Memory[(*self.shape, self.address_space, self.dtype)]
        self.fx_node.type = type_expr


@define_op("self_index")
@dataclass
class SelfIndex(CustomOp):
    dim: IndexExpr
    dtype: DataType
    elements_per_thread: Optional[IndexExpr | int] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return [self.dim]

    @property
    def type(self) -> "Register":
        return Register[(self.dim, self.dtype)]


@define_op("set_wave_prio")
@dataclass
class SetWavePrio(CustomOp):
    """
    An op that sets/tells hardware what level of priority certain instructions/region is.
    This is useful for ping-pong or general case where two Waves share the same SIMD, but
    we want to tell the SIMD to prioritize on wave or the other.
    """

    priority: int

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("shared_memory_barrier")
@dataclass
class SharedMemoryBarrier(CustomOp):
    """
    Represents a shared memory barrier in the graph.
    """

    wait_async_ops: bool = False
    tensor_wait: bool = False

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("shared_memory_barrier_signal")
@dataclass
class SharedMemoryBarrierSignal(CustomOp):
    """
    Represents a shared memory barrier signal in the graph. (gfx12)
    Argument specifies which barrier to signal.
    [1:16]:  named barriers
     0:     NOOP
    -1:     works as s_barrier
    -2:     trap barrier
    -3:     cluster barrier
    """

    barId: int = 0
    tensor_wait: bool = False

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("shared_memory_barrier_wait")
@dataclass
class SharedMemoryBarrierWait(CustomOp):
    """
    Wait for all waves in a WG to signal the barrier before proceeding. (gfx12)
    synchronize waves within a WG.
    Argument specifies which barrier to wait on.
    [1:16]:  named barriers
     0:     NOOP
    -1:     works as s_barrier
    -2:     trap barrier
    -3:     cluster barrier
    """

    barId: int = 0

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("scheduling_barrier")
@dataclass
class SchedulingBarrier(CustomOp):
    """
    Represents a scheduling barrier in the graph.
    Takes in a list of operations that are allowed to cross
    the barrier.
    """

    operations: list[Operation]


@define_op("atomic_min")
@dataclass
class AtomicOp(BinaryOpBase):
    """
    Represents an atomic operation in the graph. Takes in Register and
    Memory as inputs and writes the modified value back on to the buffer.
    Mapping attribute maps the index from wave kernel to the shared memory
    index the wavegroup operates on.
    """

    elements_per_thread: Optional[Any] = None
    mapping: Optional[IndexMapping] = None
    mapping_dynamic_vals: tuple[fx.Node, ...] = ()

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        if self.mapping is not None:
            return list(self.mapping.output_shape)
        return list(self.memory_type.symbolic_shape)

    def infer_type(self, *args):
        self.type = get_custom(self.lhs).type

    @property
    def memory_type(self) -> "Memory":
        return get_custom(self.rhs).type


@define_op("atomic_add")
@dataclass
class AtomicAddOp(AtomicOp):
    pass


@define_op("scheduling_group_barrier")
@dataclass
class SchedulingGroupBarrier(CustomOp):
    """
    Represents a scheduling group barrier in the graph.
    The scheduling group barrier defines scheduling groups.
    Each scheduling group contains different instructions in a specific order.
    The sync_id identifies scheduling groups that need to be aware of each other.
    """

    instructions: dict[Operation, int]
    sync_id: int


@define_op("memory_counter_wait")
@dataclass
class MemoryCounterWait(CustomOp):
    """
    Wait for the specified counters to be less-than or equal-to
    the provided values before continuing.
    """

    load: Optional[int] = None
    store: Optional[int] = None
    ds: Optional[int] = None
    exp: Optional[int] = None

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("workgroup_barrier")
@dataclass
class WorkgroupBarrier(CustomOp):
    """
    Represents a synchronization of all threads in a workgroup.
    Threads will wait on a WorkgroupBarrier until all the threads
    in the workgroup has called a WorkgroupBarrier(does not have to
    be in the same location).
    """

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("register")
@dataclass
class NewRegister(CustomOp):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    value: float

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return list(self.shape)

    def infer_type(self, *args):
        self.type = Register[(*self.shape, self.dtype)]


@define_op("scalar")
@dataclass
class NewScalar(CustomOp):
    value: float | IndexExpr
    dtype: DataType

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return list()

    def infer_type(self, *args):
        self.type = self.dtype


class MMABase(CustomOp):
    pass


@define_op("mma")
@dataclass
class MMA(MMABase):
    lhs: fx.Node
    rhs: fx.Node
    acc: fx.Node
    mma_type: Optional["MMAType"] | "GenericDot" = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        combined_dims = (
            get_custom(self.lhs).indexing_dims
            + get_custom(self.rhs).indexing_dims
            + get_custom(self.acc).indexing_dims
        )
        unique_dims = list(dict.fromkeys(combined_dims))
        return unique_dims

    @property
    def lhs_type(self) -> Memory:
        return get_custom(self.lhs).type

    @property
    def rhs_type(self) -> Memory:
        return get_custom(self.rhs).type

    @property
    def acc_type(self) -> Memory:
        return get_custom(self.acc).type

    def infer_type(self, *args):
        self.type = self.acc_type

    def operand_index(
        self, operand_map: dict[IndexSymbol, int], shape: list[IndexExpr]
    ) -> dict[IndexSymbol, IndexSequence]:
        from ..wave.utils.general_utils import infer_dim

        indices: dict[IndexSymbol, IndexSequence] = {}
        for dim_expr in shape:
            dim = infer_dim(dim_expr)
            indices[dim] = self.index[dim].subs(operand_map)
        return indices

    @property
    def lhs_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {MMA_LHS: 1, MMA_RHS: 0, MMA_ACC: 0}
        return self.operand_index(operand_map, self.lhs_type.symbolic_shape)

    @property
    def rhs_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {MMA_LHS: 0, MMA_RHS: 1, MMA_ACC: 0}
        return self.operand_index(operand_map, self.rhs_type.symbolic_shape)

    @property
    def acc_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}
        if self.acc_type is None:
            return None
        return self.operand_index(operand_map, self.acc_type.symbolic_shape)

    def custom_string(self, value_map: dict[str, str]) -> str:
        if self.index is None:
            return super().custom_string(value_map)
        custom_str = f"{self.tkw_op_name}("
        custom_str += f"lhs={self.lhs} (index = {self.lhs_index}), "
        custom_str += f"rhs={self.rhs} (index = {self.rhs_index}), "
        custom_str += f"acc={self.acc} (index = {self.acc_index}))"
        custom_str += f" type({self.fx_node.type})"
        return custom_str

    @property
    def reduction_dim(self) -> IndexSymbol:
        if hasattr(self.fx_node, "reduction_dim"):
            return self.fx_node.reduction_dim

    @reduction_dim.setter
    def reduction_dim(self, value: IndexSymbol):
        self.fx_node.reduction_dim = value


@define_op("scaled_mma")
@dataclass
class ScaledMMA(MMABase):
    lhs: fx.Node
    lhs_scale: fx.Node
    rhs: fx.Node
    rhs_scale: fx.Node
    acc: fx.Node
    mma_type: Optional["ScaledMMAType"] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        combined_dims = (
            get_custom(self.lhs).indexing_dims
            + get_custom(self.lhs_scale).indexing_dims
            + get_custom(self.rhs).indexing_dims
            + get_custom(self.rhs_scale).indexing_dims
            + get_custom(self.acc).indexing_dims
        )
        unique_dims = list(dict.fromkeys(combined_dims))
        return unique_dims

    @property
    def lhs_type(self) -> Memory:
        return get_custom(self.lhs).type

    @property
    def lhs_scale_type(self) -> Memory:
        return get_custom(self.lhs_scale).type

    @property
    def rhs_type(self) -> Memory:
        return get_custom(self.rhs).type

    @property
    def rhs_scale_type(self) -> Memory:
        return get_custom(self.rhs_scale).type

    @property
    def acc_type(self) -> Memory:
        return get_custom(self.acc).type

    def infer_type(self, *args):
        self.type = self.acc_type

    def operand_index(
        self, operand_map: dict[IndexSymbol, int], shape: list[IndexExpr]
    ) -> dict[IndexSymbol, IndexSequence]:
        from ..wave.utils.general_utils import infer_dim

        indices: dict[IndexSymbol, IndexSequence] = {}
        for dim_expr in shape:
            dim = infer_dim(dim_expr)
            indices[dim] = self.index[dim].subs(operand_map)
        return indices

    @property
    def lhs_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {
            MMA_LHS: 1,
            MMA_RHS: 0,
            MMA_ACC: 0,
            MMA_LHS_SCALE: 0,
            MMA_RHS_SCALE: 0,
        }
        return self.operand_index(operand_map, self.lhs_type.symbolic_shape)

    @property
    def lhs_scale_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {
            MMA_LHS: 0,
            MMA_RHS: 0,
            MMA_ACC: 0,
            MMA_LHS_SCALE: 1,
            MMA_RHS_SCALE: 0,
        }
        return self.operand_index(operand_map, self.lhs_scale_type.symbolic_shape)

    @property
    def rhs_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {
            MMA_LHS: 0,
            MMA_RHS: 1,
            MMA_ACC: 0,
            MMA_LHS_SCALE: 0,
            MMA_RHS_SCALE: 0,
        }
        return self.operand_index(operand_map, self.rhs_type.symbolic_shape)

    @property
    def rhs_scale_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {
            MMA_LHS: 0,
            MMA_RHS: 0,
            MMA_ACC: 0,
            MMA_LHS_SCALE: 0,
            MMA_RHS_SCALE: 1,
        }
        return self.operand_index(operand_map, self.rhs_scale_type.symbolic_shape)

    @property
    def acc_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {
            MMA_LHS: 0,
            MMA_RHS: 0,
            MMA_ACC: 1,
            MMA_LHS_SCALE: 0,
            MMA_RHS_SCALE: 0,
        }
        if self.acc_type is None:
            return None
        return self.operand_index(operand_map, self.acc_type.symbolic_shape)

    def custom_string(self, value_map: dict[str, str]) -> str:
        if self.index is None:
            return super().custom_string(value_map)
        custom_str = f"{self.tkw_op_name}("
        custom_str += f"lhs={self.lhs} (index = {self.lhs_index}), "
        custom_str += f"lhs_scale={self.lhs_scale} (index = {self.lhs_scale_index}), "
        custom_str += f"rhs={self.rhs} (index = {self.rhs_index}), "
        custom_str += f"rhs_scale={self.rhs_scale} (index = {self.rhs_scale_index}), "
        custom_str += f"acc={self.acc} (index = {self.acc_index}))"
        custom_str += f" type({self.fx_node.type})"
        return custom_str

    def align_index(self, constraints: list["Constraint"]) -> None:
        # Local import to break circular dep.
        from ..wave.utils.general_utils import align_index_vars

        self.index = align_index_vars(self.index, constraints)

    @property
    def reduction_dim(self) -> IndexSymbol:
        if hasattr(self.fx_node, "reduction_dim"):
            return self.fx_node.reduction_dim

    @reduction_dim.setter
    def reduction_dim(self, value: IndexSymbol):
        self.fx_node.reduction_dim = value


def get_shape_from_bindings(
    constraints: list["Constraint"], target: tuple[IndexExpr]
) -> list[IndexExpr]:
    from ..wave.utils.general_utils import get_iterator_bindings

    bindings = get_iterator_bindings(constraints)
    assert bindings is not None, "No iterator bindings found"
    bindings = bindings.bindings
    if any(dim not in bindings.keys() for dim in target):
        raise ValueError(f"Target dims {target} not found in bindings {bindings}")
    return [bindings[dim] for dim in target]


@define_op("read")
@dataclass
class Read(CustomOp):
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None
    mapping: Optional[IndexMapping] = None
    mapping_dynamic_vals: tuple[fx.Node, ...] = ()
    bounds: Optional[dict[IndexSymbol, IndexExpr]] = None
    source: Optional[tuple[IndexExpr]] = None
    target: Optional[tuple[IndexExpr]] = None
    _write_dependency: Optional[list[fx.Node]] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        from ..wave.utils.general_utils import infer_dim

        if self.mapping is not None:
            return list(self.mapping.output_shape)
        # TODO: This could contain ints.
        shape = list(self.memory_type.symbolic_shape)
        dims = [infer_dim(expr) for expr in shape]
        return dims

    def infer_type(self, *args):
        from ..wave.utils.general_utils import infer_dim

        dtype = self.memory_type.dtype
        memory_shape = list(self.memory_type.symbolic_shape)
        dim_to_shape = {infer_dim(expr): expr for expr in memory_shape}
        # Sub in dim's value from memory's type into indexing dim who contains the
        # correct order/mapping/tensor dims for dst type.
        shape = [dim_to_shape.get(dim, dim) for dim in self.indexing_dims]
        # Override when using new index mapping API.
        if self.target:
            shape = get_shape_from_bindings(args[0], self.target)
        self.type = Register[(*shape, dtype)]

    @property
    def memory_type(self) -> "Memory":
        return get_custom(self.memory).type

    @property
    def dtype(self) -> DataType:
        return self.memory_type.dtype

    @property
    def write_dependency(self) -> fx.Node:
        return self._write_dependency

    @write_dependency.setter
    def write_dependency(self, value: fx.Node):
        self.update_arg("_write_dependency", value)

    def transform_index_backwards(
        self, index: dict[IndexSymbol, IndexSequence], arg: fx.Node
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Propagate index backwards.

        Dynamic values potentially can have non-identity mapping, so we need
        to update index when walking from the node to dyn val arguments.

        E.g. if `index` is $idx and dynamic_val_mappings={N: j // ELEMS_PER_THREAD}
        resulted arg index will be $idx // ELEMS_PER_THREAD.
        """
        if arg in self.mapping_dynamic_vals:
            assert self.mapping.is_output_identity()
            i = self.mapping_dynamic_vals.index(arg)
            iters = self.mapping.iters
            mapping = self.mapping.dynamic_val_mappings[i]

            # This logic assumes that the output mapping is identity.
            subs = {
                k: index[v] for k, v in zip(iters, self.mapping.output_mapping.keys())
            }
            return {
                k: IndexSequence.from_expr(mapping[k], subs)
                for k in get_custom(arg).type.symbolic_shape
                if k in mapping
            }

        return index

    def get_derived_indices(
        self,
    ) -> list[tuple[dict[IndexSymbol, IndexSequence], fx.Node]]:
        def transform_idx(arg):
            # Treat zero index as 'not-set' and does't propagate it.
            # TODO: `set_thread_independent_index` currently blindly sets zero
            # index to all dims which are not participating in constraints, we
            # need to refactor `index_sequence_analysis` into proper dataflow
            # analysis.
            return {
                k: v
                for k, v in self.transform_index_backwards(self.index, arg).items()
                if v.start != 0
            }

        return [(arg, transform_idx(arg)) for arg in self.mapping_dynamic_vals]

    def has_identity_mapping(self) -> bool:
        """Check if mapping between input memory and output register is identity."""
        mapping = self.mapping
        if mapping is None:
            return True

        mem_shape = get_custom(self.memory).type.symbolic_shape
        if mapping.is_identity() and mapping.input_shape == mem_shape:
            return True

        return False

    def is_contiguous_vec(self, constraints, target: str) -> bool:
        """Check if op can be lowered to contiguous vector ops

        If False we will have to lower it to gather"""
        if read_meets_hw_transpose_requirements(self, constraints, target):
            return True

        if self.has_identity_mapping():
            return True

        if self.elements_per_thread == 1:
            return True

        from ..wave.utils.mapping_utils import (
            check_is_mapping_contiguous,
            check_is_dynamic_vals_broadcasted,
        )

        if not check_is_dynamic_vals_broadcasted(self.mapping_dynamic_vals):
            return False

        mapping = self.mapping

        memory = get_custom(self.memory)
        symbolic_shape = memory.type.symbolic_shape
        array_shape = symbolic_shape
        if memory.type.address_space == SHARED_ADDRESS_SPACE:
            array_shape = memory.distributed_shape

        return check_is_mapping_contiguous(
            mapping=mapping,
            symbolic_shape=symbolic_shape,
            array_shape=array_shape,
            index=self.index,
            elements_per_thread=self.elements_per_thread,
            is_read=True,
        )


class NestedRegionOp(CustomOp):
    def captured_vars(self, graph: fx.Graph) -> list[fx.Node]:
        """
        Nodes that are placeholders and are not iter args are captured vars.
        """
        captured_vars = []
        for nested_node in graph.nodes:
            custom = get_custom(nested_node)
            if isinstance(custom, Placeholder) and not isinstance(custom, IterArg):
                captured_vars.append(nested_node)
        return captured_vars

    def get_outer_node(self, outer_node: fx.Node) -> fx.Node:
        while "lifted" in outer_node.meta:
            outer_node = outer_node.meta["lifted"]
        return outer_node

    def get_captured_fx_node(
        self, graph: fx.Graph, outer_node: fx.Node
    ) -> Optional[fx.Node]:
        outer_node = self.get_outer_node(outer_node)
        for var in self.captured_vars(graph):
            custom = get_custom(var)
            if custom.get_captured_fx_node() == outer_node:
                return var

        return None

    def get_root_graph(self):
        """
        Return the "root"/outermost layer of our computation graph.
        This is done by iteratively accessing parent_graph of current
        graph. This is done until we find the "root" graph who
        will have "subgraph" attribute.
        """
        cur_graph = self.graph
        while not hasattr(cur_graph, "subgraphs"):
            if not hasattr(cur_graph, "parent_op"):
                raise ValueError("All subgraphs should have parent_op")
            cur_graph = cur_graph.parent_op.graph
        return cur_graph

    def erase(self):
        subgraphs = self.get_root_graph().subgraphs
        subgraph = subgraphs[self.subgraph_name]
        for node in list(subgraph.nodes)[::-1]:
            get_custom(node).erase()

        del subgraphs[self.subgraph_name]
        super().erase()

    def iter_args(self, graph: Optional[fx.Graph] = None) -> list[fx.Node]:
        iter_args = []
        if graph is None:
            graph = self.get_root_graph().subgraphs[self.subgraph_name]
        for nested_node in graph.nodes:
            custom = get_custom(nested_node)
            if isinstance(custom, IterArg):
                iter_args.append(nested_node)
        # Sort by iter_idx.
        iter_args = sorted(iter_args, key=lambda x: get_custom(x).iter_idx)
        return iter_args

    def outputs(self, graph: Optional[fx.Graph] = None) -> list[fx.Node]:
        if graph is None:
            graph = self.get_root_graph().subgraphs[self.subgraph_name]

        output = get_custom(graph.output_node())
        assert isinstance(output, Output), f"Expected Output, but got {output}"
        return output.return_vals[0]

    def infer_type(self, *args):
        if self.init_args is not None:
            res_types = [get_custom(x).type for x in self.init_args]
            if len(res_types) == 1:
                res_types = res_types[0]
            self.type = res_types

    @classmethod
    def handle(cls, graph: RegionGraph, *args, **kwargs):
        """
        Base handle method for nested region operations.
        Extracts tag from kwargs and sets it on the node and underlying fx.Node.
        """

        def wrapper(f):
            with graph.subtracer() as subtracer:
                subgraph_name, implicit_captures = subtracer.trace(f)
            # Extract tag from kwargs if present
            tag = kwargs.pop("tag", None)

            # For Iterate, we need to handle additional parameters
            if cls.__name__ == "Iterate":
                # Extract Iterate-specific parameters
                step = kwargs.pop("step", 1)
                start = kwargs.pop("start", None)
                condition = kwargs.pop("condition", None)
                node = cls(
                    *args,
                    **kwargs,
                    step=step,
                    start=start,
                    condition=condition,
                    subgraph_name=subgraph_name,
                    implicit_captures=implicit_captures,
                )
            else:
                # For other nested region ops like Conditional
                node = cls(
                    *args,
                    **kwargs,
                    subgraph_name=subgraph_name,
                    implicit_captures=implicit_captures,
                )

            node._add_proxy_to_graph(graph)
            node.fx_node.node.tkw_op = cls
            node.fx_node.node.tkw_op_name = cls.tkw_op_name
            node.fx_node.node.location = capture_location(graph.location_capture_config)
            if tag is not None:
                node.fx_node.node.tag = tag

            # Remember which placeholders are init args. This connection gets
            # lost otherwise
            for nested_node in graph.subgraphs[subgraph_name].nodes:
                if nested_node.op == "placeholder":
                    if nested_node not in [
                        var.node
                        for var in graph.inner_freevars[graph.subgraphs[subgraph_name]]
                    ]:
                        nested_node.tkw_op = IterArg

            graph.subgraphs[subgraph_name].parent_op = node.fx_node.node
            return node.fx_node

        return wrapper


@define_op("conditional")
@dataclass
class Conditional(NestedRegionOp):
    """
    The optional `else_return` argument must match the type of any `return` statements in the conditional body.
    The `else_return` are the default return value for the `else` branch.
    """

    condition: fx.Proxy | IndexExpr
    subgraph_name: str
    implicit_captures: Sequence[fx.Proxy]
    else_return: Optional[Sequence["Register"]] = None

    @property
    def init_args(self):
        """Alias for else_return to match Iterate interface for shared processing code."""
        return self.else_return

    @init_args.setter
    def init_args(self, value):
        self.else_return = value

    @property
    def indexing_dims(self) -> list[IndexSymbol] | list[list[IndexSymbol]]:
        # If we have else_return, the conditional returns values
        # and we need to get indexing dims from the return values
        if self.else_return:
            expand_dims: list[IndexSymbol] = []
            subgraph = self.get_root_graph().subgraphs[self.subgraph_name]
            return_node = get_custom(subgraph.output_node())
            assert isinstance(return_node, Output)
            return_vals = return_node.return_vals[0]
            if not isinstance(return_vals, Sequence):
                return_vals = [return_vals]
            for return_val in return_vals:
                return_dims = get_custom(return_val).indexing_dims
                expand_dims.append(return_dims)
            if len(expand_dims) == 1:
                expand_dims = expand_dims[0]
            return expand_dims
        # Otherwise, get from condition
        elif isinstance(self.condition, fx.Node):
            return get_custom(self.condition).indexing_dims
        return []


@define_op("iterate")
@dataclass
class Iterate(NestedRegionOp):
    axis: IndexSymbol
    init_args: Sequence[Any]
    subgraph_name: str
    implicit_captures: Sequence[fx.Proxy]
    step: int = 1
    start: Optional[IndexExpr] = None
    condition: Optional[IndexExpr] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol] | list[list[IndexSymbol]]:
        expand_dims: list[IndexSymbol] = []
        subgraph = self.get_root_graph().subgraphs[self.subgraph_name]
        return_node = get_custom(subgraph.output_node())
        assert isinstance(return_node, Output)
        return_vals = return_node.return_vals[0]
        if not isinstance(return_vals, Sequence):
            return_vals = [return_vals]
        for return_val in return_vals:
            return_dims = get_custom(return_val).indexing_dims
            reduced_dims = [dims for dims in return_dims if dims != self.axis]
            expand_dims.append(reduced_dims)
        if len(expand_dims) == 1:
            expand_dims = expand_dims[0]
        return expand_dims

    @property
    def index(self) -> list[dict[IndexSymbol, IndexSequence]]:
        subgraph = self.get_root_graph().subgraphs[self.subgraph_name]
        output = get_custom(subgraph.output_node())
        assert isinstance(output, Output)
        return_vals = output.return_vals[0]
        return (
            [
                (
                    get_custom(val).acc_index
                    if isinstance(get_custom(val), (MMA, ScaledMMA))
                    else val.index
                )
                for val in return_vals
            ]
            if isinstance(return_vals, (Sequence))
            else return_vals.index
        )

    @index.setter
    def index(self, value: Any):
        CustomOp.index.fset(self, value)

    @property
    def count(self) -> Optional[int]:
        if hasattr(self.fx_node, "count"):
            return self.fx_node.count
        return None

    @count.setter
    def count(self, value: int):
        self.fx_node.count = value


@define_op("write")
@dataclass
class Write(CustomOp):
    register_: fx.Proxy
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None
    mapping: Optional[IndexMapping] = None
    mapping_dynamic_vals: tuple[fx.Node, ...] = ()
    bounds: Optional[dict[IndexSymbol, IndexExpr]] = None
    source: Optional[tuple[IndexExpr]] = None
    target: Optional[tuple[IndexExpr]] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        from ..wave.utils.general_utils import infer_dim

        if self.mapping is not None:
            return list(self.mapping.input_shape)
        # TODO: This could contain ints.
        shape = list(self.memory_type.symbolic_shape)
        dims = [infer_dim(expr) for expr in shape]
        return dims

    def infer_type(self, *args):
        from ..wave.utils.general_utils import infer_dim

        address_space = self.memory_type.address_space
        dtype = self.memory_type.dtype
        memory_shape = list(self.memory_type.symbolic_shape)
        dim_to_shape = {infer_dim(expr): expr for expr in memory_shape}
        # Sub in dim's value from memory's type into indexing dim who contains the
        # correct order/mapping/tensor dims for dst type.
        shape = [dim_to_shape.get(dim, dim) for dim in self.indexing_dims]
        # Override when using new index mapping API.
        if self.source:
            shape = get_shape_from_bindings(args[0], self.source)
        self.type = Memory[(*shape, address_space, dtype)]

    @property
    def memory_type(self) -> "Memory":
        return get_custom(self.memory).type

    @property
    def register_type(self) -> "Register":
        return get_custom(self.register_).type

    @property
    def register_index(self) -> dict[IndexSymbol, IndexSequence]:
        custom = get_custom(self.register_)
        return custom.index

    def transform_index_backwards(
        self, index: dict[IndexSymbol, IndexSequence], arg: fx.Node
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Propagate index backwards.

        Dynamic values potentially can have non-identity mapping, so we need
        to update index when walking from the node to dyn val arguments.

        E.g. if `index` is $idx and dynamic_val_mappings={N: j // ELEMS_PER_THREAD}
        resulted arg index will be $idx // ELEMS_PER_THREAD.
        """
        if arg in self.mapping_dynamic_vals:
            assert self.mapping.is_input_identity()
            i = self.mapping_dynamic_vals.index(arg)
            iters = self.mapping.iters
            mapping = self.mapping.dynamic_val_mappings[i]

            # This logic assumes that the input mapping is identity.
            subs = {
                k: index[v] for k, v in zip(iters, self.mapping.input_mapping.keys())
            }
            return {
                k: IndexSequence.from_expr(mapping[k], subs)
                for k in arg.type.symbolic_shape
                if k in mapping
            }

        return index

    def get_derived_indices(
        self,
    ) -> list[tuple[dict[IndexSymbol, IndexSequence], fx.Node]]:
        def transform_idx(arg):
            return {
                k: v
                for k, v in self.transform_index_backwards(self.index, arg).items()
                if v.start != 0
            }

        return [(arg, transform_idx(arg)) for arg in self.mapping_dynamic_vals]

    def has_identity_mapping(self) -> bool:
        """Check if mapping between input register and output memory is identity."""
        mapping = self.mapping
        if mapping is None:
            return True

        mem_shape = get_custom(self.memory).type.symbolic_shape
        if mapping.is_identity() and mapping.output_shape == mem_shape:
            return True

        return False

    def is_contiguous_vec(self, constraints, target: str) -> bool:
        """Check if op can be lowered to contiguous vector ops

        If False we will have to lower it to gather"""
        if self.has_identity_mapping():
            return True

        if self.elements_per_thread == 1:
            return True

        from ..wave.utils.mapping_utils import (
            check_is_mapping_contiguous,
            check_is_dynamic_vals_broadcasted,
        )

        if not check_is_dynamic_vals_broadcasted(self.mapping_dynamic_vals):
            return False

        mapping = self.mapping

        memory = get_custom(self.memory)
        symbolic_shape = memory.type.symbolic_shape
        array_shape = symbolic_shape
        if memory.type.address_space == SHARED_ADDRESS_SPACE:
            array_shape = memory.distributed_shape

        return check_is_mapping_contiguous(
            mapping=mapping,
            symbolic_shape=symbolic_shape,
            array_shape=array_shape,
            index=self.index,
            elements_per_thread=self.elements_per_thread,
            is_read=False,
        )


@define_op("debug_log")
@dataclass
class DebugLog(CustomOp):
    """
    An op for debugging.
    Represents a write to an implicit global memory location.
    The kernel will implicitly have an extra memory input added that will be injected by the Python kernel launcher.
    The memory can be accessed by passing an an extra keyword to kernel invokation "debug_logs" with an empty dictionary.
    The dictionary will be mutated, adding the `label`s as keys that map to nested dictionaries.
    The nested dictionaries have a `value` field with the log tensor, and other keys (eg. `symbolic_shape`).

    IE the debug_logs dictionary will look like:
    `{LABEL: {"value": LOG_TENSOR, (other metadata keys) ...}}``

    Note that the logs collected in the `debug_logs` field, or handled by `printer` or `handler` represent a global view of the log after all writes, not limited to any one wave or loop iteration.

    The optional `printer` argument should be a function that accepts a string (the log's `label`) and the value of the log itself (a Torch tensor).
    The default value for this is `print`, though note that it will probably print an abbreviated view of the global tensor.

    The optional `handler` argument should be a function that accepts the whole `debug_logs` object (IE all logs, not just one).
    The handler function gives a way to specify something like a viewer for all logs, but specify it inline among print functions rather than separately.

    The optional `extra_iteration_dimensions` argument allows you to add extra dimensions to capture values from multiple iterations of a loop.
    It takes a list of tuples, where each tuple contains `(dimension_name, iteration_axis, max_iterations)`.
    The `dimension_name` must be a unique symbol, and will be the name of the dimension in the symbolic shape of the output.
    The `iteration_axis` must be the axis of an `Iterate` operation, IE the dimension being reduced in the iteration.
    The `max_iterations` argument is a positive integer that represents the maximum number of iterations to store.
    Note that if you give too few, later iterations will currently overwrite the final iteration slot.

    The API and semantics of this operation are not yet stable, but since it is just a debugging tool, you want to take any debug logging out of your kernel before shipping it anyway.
    """

    register_: fx.Proxy
    label: Optional[str] = None
    extra_iteration_dimensions: Optional[
        list[tuple["IndexSymbol", "IndexSymbol", int]]
    ] = None
    mapping: Optional[IndexMapping] = None
    mapping_dynamic_vals: tuple[fx.Node, ...] = ()
    printer: Optional[Callable[[str, Any], Any]] = print
    handler: Optional[Callable[[dict[str, Any]], Any]] = None

    @property
    def memory(self) -> Optional[fx.Proxy]:
        return self.fx_node.memory

    @memory.setter
    def memory(self, value):
        self.fx_node.memory = value

    @property
    def has_side_effects(self) -> bool:
        return True

    def infer_type(self, *args):

        regtype = get_custom(self.register_).type
        mem = get_custom(self.memory)
        type_expr = Memory[regtype.symbolic_shape, GLOBAL_ADDRESS_SPACE, regtype.dtype]
        mem.type = type_expr

        self.type = type_expr
        self.fx_node.type = type_expr


@define_op("bounds_check")
@dataclass
class BoundsCheck(CustomOp):
    """
    Checks if a dimension is within a bound.
    """

    index_exprs: dict[IndexExpr, IndexSequence]
    bounds: dict[IndexSymbol, IndexExpr]
    mapping: Optional[IndexMapping] = None
    mapping_dynamic_vals: tuple[fx.Node, ...] = ()
    mask_bounds: Optional[dict[IndexSymbol, IndexExpr]] = None

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("apply_expr")
@dataclass
class ApplyExpr(CustomOp):
    register_: fx.Proxy | Sequence[fx.Proxy]
    expr: Callable

    @property
    def type(self) -> "Register":
        reg = self.register_
        if not isinstance(reg, Sequence):
            return get_custom(reg).type

        types = [get_custom(r).type for r in reg]
        assert len(set(types)) == 1, f"Types mismatch: {types}"
        return types[0]

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        reg = self.register_
        if not isinstance(reg, Sequence):
            return get_custom(reg).indexing_dims

        dims = [tuple(get_custom(r).indexing_dims) for r in reg]
        assert len(set(dims)) == 1, f"Indexing dims mismatch: {dims}"
        return dims[0]


@define_op("set_symbol")
@dataclass
class SetSymbol(CustomOp):
    symbol: IndexExpr
    register_: fx.Proxy

    @property
    def type(self) -> "Register":
        return get_custom(self.register_).type

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.register_).indexing_dims

    @property
    def has_side_effects(self) -> bool:
        return True


@define_py_op(operator.getitem)
@define_op("get_result")
@dataclass
class GetResult(CustomOp):
    value: fx.Node
    res_idx: int

    def infer_type(self, *args):
        op = get_custom(self.value)
        src_type = op.type
        if isinstance(src_type, list):
            if self.res_idx >= len(src_type):
                raise RuntimeError(
                    f"GetResult of {self.res_idx} from result with {len(src_type)} results"
                    f"\n{op=}\nsrc={self.value}\n{src_type=}"
                )
            self.type = src_type[self.res_idx]
        else:
            self.type = src_type

    @property
    def indexing_dims(self) -> list[IndexExpr]:
        has_multiple_value = lambda x: all(isinstance(el, list) for el in x)
        is_valid_indexing_dim = lambda x: isinstance(src_indexing, list) and all(
            isinstance(el, IndexExpr) for el in x
        )
        src_indexing = get_custom(self.value).indexing_dims
        if has_multiple_value(src_indexing):
            assert self.res_idx < len(src_indexing), f"{self=}"
            src_indexing = src_indexing[self.res_idx]
        assert is_valid_indexing_dim(src_indexing)
        return src_indexing

    @property
    def index(self) -> dict[IndexSymbol, IndexSequence]:
        custom = get_custom(self.value)
        custom_index = custom.index
        if custom_index is None:
            return None
        if not isinstance(custom, Iterate):
            return custom_index
        if not isinstance(custom_index, Sequence):
            return custom_index

        # `indexing_dims` is way too expensive to use it in assert.
        # assert self.res_idx < len(
        #     custom.indexing_dims
        # ), f"Invalid {custom_index=} with {self.res_idx=} and {custom.indexing_dims=}\n{custom}"
        return custom_index[self.res_idx]

    @index.setter
    def index(self, value: dict[IndexSymbol, IndexSequence]):
        CustomOp.index.fset(self, value)

    @property
    def distributed_shape(self):
        iterate = get_custom(self.value)
        allocate = get_custom(iterate.init_args[self.res_idx])
        assert isinstance(allocate, Allocate)
        return allocate.distributed_shape


@define_op("extract")
@dataclass
class Extract(CustomOp):
    """
    Op Rationale:

    Extract is an op used to represent extracting of
    a scalar from TKW's 1-D vector on the specified index.

    This can also be viewed as indexing/slicing on the fastest
    dimension. Hence, the semantic of this op is designed to
    see itself as a reduction on the indexed/fastest dimension.
    """

    register_: fx.Proxy
    offset: IndexExpr | int

    def infer_type(self, *args):
        # Intuition here is we are trying to extract an element
        # from fastest dim => we reduce the fastest dim.
        src_type = get_custom(self.register_).type
        # Return itself if just 0-D/1-D symbolic.
        if len(src_type.symbolic_shape) <= 1:
            self.type = src_type
            return

        # Typically fastest dim is the last dimension,
        # If non-unit dim exists => non-unit dim is fastest dim.
        non_unit_dim = [k for k, v in self.register_.index.items() if v.size != 1]
        if len(non_unit_dim) > 1:
            raise NotImplementedError(
                f"NYI: Extract only support 1 non-unit dim, but found: {len(non_unit_dim)}"
            )
        dst_shape = list(src_type.symbolic_shape)
        dim_to_remove = dst_shape[-1] if not non_unit_dim else non_unit_dim[0]
        dst_shape.remove(dim_to_remove)
        dst_type = Register[(*dst_shape, src_type.dtype)]
        self.type = dst_type


@define_op("extract_slice")
@dataclass
class ExtractSlice(CustomOp):
    register_: fx.Proxy
    offset: tuple[IndexExpr]
    size: tuple[IndexExpr]
    stride: tuple[IndexExpr]

    @property
    def type(self) -> "Register":
        return get_custom(self.register_).type

    @property
    def rank(self) -> int:
        offset_rank = len(self.offset)
        size_rank = len(self.size)
        stride_rank = len(self.stride)
        assert (
            offset_rank == size_rank == stride_rank
        ), "Expected offset, size, and stride to have same rank."
        return size_rank


@define_op("broadcast")
@dataclass
class Broadcast(CustomOp, ABC):
    """
    Represents a Broadcast operation.

    arg: Source tensor/value to broadcast
    target_shape: symbolic target broadcast shape.
    """

    arg: fx.Node
    target_shape: Sequence[IndexSymbol] = None

    def __post_init__(self):
        # Required for setting up hash.
        super().__post_init__()
        # Verify for valid src type.
        if isinstance(self.arg, fx.Node):
            src = self.arg
        elif isinstance(self.arg, fx.Proxy):
            src = self.arg.node
        else:
            raise ValueError(f"Unexpected broadcast src type of {type(self.arg)}")

        # Verifies target broadcast shape is valid.
        src_type = get_custom(src).type
        src_shape = set(getattr(src_type, "symbolic_shape", []))
        dst_shape = set(self.target_shape)
        assert src_shape.issubset(
            dst_shape
        ), "Fail to initialize broadcast because of invalid target_shape."

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        from ..wave.utils.general_utils import infer_dim

        # infer_dim to handle cases where target_shape has
        # scaled expressions (i.e K/32 -> K).
        dims = [infer_dim(expr) for expr in self.target_shape]
        return dims

    def infer_type(self, *args):
        src_dtype = get_custom(self.arg).type.dtype
        self.type = Register[(*self.target_shape, src_dtype)]


@define_interface_op("cumsum")
@dataclass
class ScanOp(CustomOp, ABC):
    """
    Base class for all scan-style operations (e.g., cumsum).

    arg: Source tensor/value to scan.
    init: Optional initial value.
    dim: Symbolic dimension along which to scan.
    """

    arg: fx.Node | list[fx.Node]
    init: Optional[fx.Node] = None
    dim: Optional[IndexSymbol] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        from ..wave.utils.general_utils import all_equal

        if isinstance(self.arg, Sequence):
            src_indexings = [get_custom(arg).indexing_dims for arg in self.arg]
            if not all_equal(src_indexings):
                raise NotImplementedError(
                    "All inputs to ScanOp must have same indexing dims."
                )
            indexing = src_indexings[0]
        else:
            indexing = get_custom(self.arg).indexing_dims

        return [dim for dim in indexing if dim != self.dim]

    def infer_type(self, *args):
        if isinstance(self.arg, Sequence):
            src_types = [get_custom(arg).type for arg in self.arg]
            ref_shape = src_types[0].symbolic_shape
            ref_dtype = src_types[0].dtype
            for src_type in src_types:
                if src_type.symbolic_shape != ref_shape or src_type.dtype != ref_dtype:
                    raise NotImplementedError(
                        "ScanOp requires all args to have same shape and dtype."
                    )
            src_type = src_types[0]
        else:
            src_type = get_custom(self.arg).type

        # normalize dim=-1 or None
        if self.dim == -1 or self.dim is None:
            self.dim = src_type.symbolic_shape[-1]

        self.type = Register[(*src_type.symbolic_shape, src_type.dtype)]

        if self.init is not None:
            init_shape = get_custom(self.init).type.symbolic_shape
            if init_shape != self.type.symbolic_shape:
                raise RuntimeError(
                    f"Init shape {init_shape} must match result shape {self.type.symbolic_shape}"
                )

    @property
    def scan_dim(self) -> IndexSymbol:
        return self.dim


@define_interface_op("max")
@define_interface_op("min")
@define_interface_op("sum")
@dataclass
class ReduceOp(CustomOp, ABC):
    """
    Represents a Reduce computation.

    arg: Source tensor/value to reduce
    init: init/accumulator for reduce
    dim: which symbolic dim to reduce.
    block: When set to true, reduce across block, else reduce across warp.
    """

    arg: fx.Node | list[fx.Node]
    init: fx.Node = None
    dim: Optional[Any] = None
    block: Optional[bool] = False

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        # Local import to break circular dep.
        from ..wave.utils.general_utils import all_equal

        if isinstance(self.arg, Sequence):
            src_indexings = [get_custom(arg).indexing_dims for arg in self.arg]
            if not all_equal(src_indexings):
                raise NotImplementedError(
                    "NYI: Only support case where all inputs to ReduceOp to have same indexing dim."
                )
            src_indexing = src_indexings[0]
        else:
            src_indexing = get_custom(self.arg).indexing_dims
        dst_indexing = [dim for dim in src_indexing if dim != self.dim]
        return dst_indexing

    def infer_type(self, *args):
        if isinstance(self.arg, Sequence):
            src_types = [get_custom(arg).type for arg in self.arg]
            ref_shape = src_types[0].symbolic_shape
            ref_dtype = src_types[0].dtype
            if not all(
                src_type.symbolic_shape == ref_shape and src_type.dtype == ref_dtype
                for src_type in src_types
            ):
                raise NotImplementedError(
                    "NYI: Only support case where all inputs to ReduceOp to have same type."
                )
            src_type = src_types[0]
        else:
            src_type = get_custom(self.arg).type
        dtype = src_type.dtype
        reduced_dims = [dims for dims in src_type.symbolic_shape if dims != self.dim]
        self.type = Register[(*reduced_dims, dtype)] if reduced_dims else dtype
        if (
            self.init is not None
            and get_custom(self.init).type.symbolic_shape != self.type.symbolic_shape
        ):
            raise RuntimeError(
                f"Init type for {self.tkw_op_name} {get_custom(self.init).type.symbolic_shape}"
                f" must match reduce type {self.type.symbolic_shape}"
                f"\n{self}"
            )

    @property
    def reduction_dim(self) -> IndexSymbol:
        return self.dim


# TODO: Add support for more shuffle types.
@define_op("shuffle")
@dataclass
class ShuffleOp(CustomOp):
    """
    Represents a shuffle.xor op.

    arg: value/vector to shuffle.
    offset: xor offset.
    width: xor width.
    """

    arg: fx.Node
    offset: int
    width: int
    mode: "ShuffleMode"

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    def infer_type(self, *args):
        self.type = get_custom(self.arg).type


@define_op("topk")
@dataclass
class TopkOp(CustomOp):
    """
    Represents a TopK computation.

    arg: Source tensor Register value to find top-k from
    k_dim: Dimension symbol representing the K size (number of top elements to select)
    dim: which symbolic dim to perform topk on.  This dimension will be replaced by k_dim in the output.
    """

    arg: fx.Node
    k_dim: IndexSymbol
    dim: IndexSymbol

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        from ..wave.utils.general_utils import all_equal

        if isinstance(self.arg, Sequence):
            src_indexings = [get_custom(arg).indexing_dims for arg in self.arg]
            if not all_equal(src_indexings):
                raise NotImplementedError(
                    "NYI: Only support case where all inputs to TopkOp to have same indexing dim."
                )
            src_indexing = src_indexings[0]
        else:
            src_indexing = get_custom(self.arg).indexing_dims

        # TopK replaces the reduction dimension with the K dimension
        if self.dim in src_indexing:
            result_dims = []
            for dim in src_indexing:
                if dim == self.dim:
                    result_dims.append(self.k_dim)
                else:
                    result_dims.append(dim)
            return result_dims
        return src_indexing

    def infer_type(self, *args):
        src_type = get_custom(self.arg).type
        dtype = src_type.dtype
        src_indexing = get_custom(self.arg).indexing_dims

        new_symbolic_shape = []
        for i, dim_size in enumerate(src_type.symbolic_shape):
            if i < len(src_indexing) and src_indexing[i] == self.dim:
                new_symbolic_shape.append(self.k_dim)
            else:
                new_symbolic_shape.append(dim_size)

        # TopK returns tuple of (values, indices) with the new shape
        # Values have the same dtype as input, indices are i32
        values_type = Register[(*new_symbolic_shape, dtype)]
        indices_type = Register[(*new_symbolic_shape, i32)]
        self.type = [values_type, indices_type]

    @property
    def reduction_dim(self) -> IndexSymbol:
        return self.dim

    # Provide init field to be compatible with ReduceOp interface.
    # This allows some code in the expand_graph pass to be generic over the two.
    @property
    def init(self):
        return None


@define_op("cast")
@dataclass
class CastOp(CustomOp, ABC):
    """
    Represents a cast operation.
    """

    arg: fx.Node
    dtype: DataType

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    def infer_type(self, *args):
        src_shape = get_custom(self.arg).type.symbolic_shape
        self.type = Register[(*src_shape, self.dtype)]


@define_op("bitcast")
@dataclass
class BitcastOp(CustomOp, ABC):
    """
    Represents a bitcast operation.
    """

    arg: fx.Node
    dtype: DataType

    @property
    def scale_factor(self):
        src_width = self.arg.type.dtype.bitwidth()
        dst_width = self.dtype.bitwidth()
        if src_width % dst_width != 0:
            raise NotImplementedError(
                "Currently only support bitcast if src_width % dst_width == 0."
            )
        return int(src_width / dst_width)

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    def infer_type(self, *args):
        src_shape = get_custom(self.arg).type.symbolic_shape
        dst_shape = list(src_shape)
        dst_shape[-1] *= self.scale_factor
        self.type = Register[(*dst_shape, self.dtype)]


@define_op("permute")
@dataclass
class Permute(CustomOp, ABC):
    """
    Represents a permute operation that
    permutes arg into the target shape.
    """

    arg: fx.Node
    target_shape: Sequence[IndexExpr]

    @property
    def indexing_dims(self) -> list[IndexExpr]:
        return self.target_shape

    def infer_type(self, *args):
        src_type = get_custom(self.arg).type
        assert set(src_type.symbolic_shape) == set(
            self.target_shape
        ), f"Target shape {self.target_shape} must be a permutation of source shape {src_type.symbolic_shape}"
        self.type = Register[(*self.target_shape, src_type.dtype)]

    def transform_index(
        self, index: dict[IndexSymbol, IndexSequence]
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        The permute operation swaps the strides of the permuted indices.
        So say we have a permute operation that swaps [B, M, N] to
        [M, N, B], then we swap the strides of the dimensions.
        """
        # Assert if `vector_shapes` is unknown
        assert (
            self.vector_shapes is not None
        ), "`vector_shapes` must be set before calling this function"

        custom_src = get_custom(self.arg)
        src_shape = custom_src.type.symbolic_shape
        src_to_target = {
            src: self.target_shape[src_shape.index(src)] for src in src_shape
        }
        # Compute the non-unit iterators, ie. where `vector_shapes != 0`
        non_unit_its = [k for k, v in self.vector_shapes.items() if v != 0]
        # Compute the non-unit shapes, both source and target
        non_unit_src = [d for d in src_shape if d in non_unit_its]
        non_unit_tgt = [d for d in self.target_shape if d in non_unit_its]
        # If `non_unit_src == non_unit_tgt` permute only the indices of the op.
        # For example, if `src_shape == [B, M, N]`, `tgt_shape = [M, B, N]`,
        # and `vector_shape(B) == 0`, then `non_unit_src = [M, N]`,
        # `non_unit_tgt = [M, N]`. Meaning only trivial operators are being
        # permuted, thus only the order of the indices must be updated.
        if non_unit_src == non_unit_tgt:
            return {k: index[k] for k in self.target_shape if k in index}
        # Else, permute the strides.
        permuted_index = {
            k: IndexSequence(v.start, v.size, index[src_to_target[k]].stride)
            for k, v in index.items()
            if k in src_shape
        }
        return permuted_index


def _to_sequence(input: Any | Sequence[Any]) -> Sequence[Any]:
    return input if isinstance(input, Sequence) else (input,)


@define_op("reshape")
@dataclass
class Reshape(CustomOp, ABC):
    """
    Represents a reshape operation that reshapes
    vectors along the same dimension.

    """

    args: fx.Node | Sequence[fx.Node]
    target_vector_shape: dict[IndexSymbol, int]

    @property
    def indexing_dims(self) -> list[IndexExpr]:
        return get_custom(_to_sequence(self.args)[0]).indexing_dims

    def infer_type(self, *args):
        self.type = get_custom(_to_sequence(self.args)[0]).type


@define_op("tensor_load_to_lds")
@dataclass
class TensorLoadToLDS(CustomOp):
    src: list[Memory]
    dst: list[Memory]
    element_type: DataType
    distributed_shape: dict[IndexSymbol, IndexExpr]
    shared_tile_index: dict[IndexSymbol, IndexSequence]
    global_tile_index: dict[IndexSymbol, IndexSequence]
    bounds: dict[IndexSymbol, IndexExpr]
    multicast_mask: Optional[IndexExpr] = None
    input_selector: IndexSymbol | int = 0

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("gather_to_lds")
@dataclass
class GatherToLDS(CustomOp):
    """
    Represents an instruction that performs direct load from global
    to lds. Source node points to the global memory to load from
    and the destination node points to shared memory.
    """

    src: Memory
    dst: Memory
    src_index: dict[IndexSymbol, IndexSequence]
    dst_index: dict[IndexSymbol, IndexSequence]
    dtype: DataType
    elements_per_thread: Optional[IndexExpr | int]
    src_mapping: Optional[IndexMapping]
    dst_mapping: Optional[IndexMapping]
    src_bounds: Optional[dict[IndexSymbol, IndexExpr]]
    src_mapping_dynamic_vals: tuple[fx.Node, ...] = ()
    dst_mapping_dynamic_vals: tuple[fx.Node, ...] = ()

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("scatter_add")
@dataclass
class ScatterAdd(CustomOp):
    """
    ScatterAdd performs element-wise accumulation from a source register into shared memory (LDS),
    at locations determined by the index register along a specified dimension.

    Limitations:
    - Only intra-workgroup scattering is supported (i.e., within shared memory / LDS), assuming a single wave.
    - Multi-wave execution is not guaranteed to be safe: synchronization issues may occur when threads write to the same index. Further investigation is needed.
    - The operation supports multiple elements per thread, assuming the non-scatter dimension is large enough (i.e., > elements_per_thread).
    """

    register_src: fx.Node
    register_idx: fx.Node
    dim: IndexExpr
    memory: fx.Node
    mapping: IndexMapping
    elements_per_thread: Optional[int] = 1
    bounds: Optional[dict[IndexSymbol, IndexExpr]] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        if self.mapping is not None:
            return list(self.mapping.input_shape)
        return list(self.memory_type.symbolic_shape)

    def infer_type(self, *args):
        address_space = self.memory_type.address_space
        dtype = self.memory_type.dtype
        self.type = Memory[(*self.indexing_dims, address_space, dtype)]

    @property
    def memory_type(self) -> "Memory":
        return get_custom(self.memory).type

    @property
    def register_type(self) -> "Register":
        return get_custom(self.register_src).type

    @property
    def register_index(self) -> dict[IndexSymbol, IndexSequence]:
        custom = get_custom(self.register_src)
        return custom.index

    @property
    def has_side_effects(self) -> bool:
        return True
