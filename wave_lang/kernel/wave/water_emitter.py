#!/usr/bin/env python3
"""
Water Emitter for Wave Dialect

This generates operations Wave dialect from a serialized wave trace (json).
It runs as a standalone process with access to Water Python bindings and emits
Wave operations with fallback support for unknown operations.
"""

import json
import sys

try:
    from water_mlir import ir
    from water_mlir.water_mlir.dialects.wave import (
        AddOp,
        DivOp,
        Exp2Op,
        MulOp,
        ReadOp,
        RegisterOp,
        WriteOp,
    )
    from water_mlir.water_mlir.dialects import wave
except Exception as e:
    print(f"FATAL: failed to import water_mlir: {e}", file=sys.stderr)
    sys.exit(2)

# Mapping from tkw_op_name to actual op constructors
WAVE_OP_CONSTRUCTORS = {
    "add": AddOp,
    "mul": MulOp,
    "div": DivOp,
    "exp2": Exp2Op,
    "read": ReadOp,
    "write": WriteOp,
    "register": RegisterOp,
    # TODO: Add more or find a good way of avoiding needing a mapping
}


def _default_tensor_type():
    # Return a default Wave tensor type for register operations
    return ir.Type.parse("!wave.tensor<any of f32, <register>>")


def _default_memory_tensor_type():
    # Return a default Wave tensor type for memory operations
    return ir.Type.parse("!wave.tensor<any of f32, <global>>")


def _map_address_space(addr: str) -> str:
    if not addr:
        return ""
    a = str(addr).lower()
    if "global" in a:
        return "global"
    if "shared" in a:
        return "shared"
    if "register" in a:
        return "register"
    # If we can't determine the address space, use unspecified
    return "unspecified"


def _build_wave_type_text(wave_type_dict: dict) -> str:
    kind = wave_type_dict.get("kind")
    shape = wave_type_dict.get("shape", []) or []
    dtype = wave_type_dict.get("dtype", "f32")
    addr = wave_type_dict.get("address_space", "")
    if not shape:
        shape_txt = "any"
    else:
        shape_txt = ", ".join([f"@{s}" for s in shape])
    addr_norm = _map_address_space(addr)
    addr_txt = f", <{addr_norm}>" if addr_norm else ""
    return f"!wave.tensor<[{shape_txt}] of {dtype}{addr_txt}>"


def _type_from_wave_type_dict(ctx: ir.Context, wave_type_dict: dict) -> ir.Type:
    # Use textual parser to construct the Wave type.
    type_text = _build_wave_type_text(wave_type_dict)
    return ir.Type.parse(type_text, context=ctx)


def create_operation_attributes(node_data):
    """Create MLIR attributes from node data."""
    attrs = {}

    # Map common attributes
    if "name" in node_data:
        attrs["fx.name"] = ir.StringAttr.get(str(node_data["name"]))
    if "op" in node_data:
        attrs["fx.op"] = ir.StringAttr.get(str(node_data["op"]))
    if "tkw_op_name" in node_data:
        attrs["fx.tkw_op_name"] = ir.StringAttr.get(str(node_data["tkw_op_name"]))
    if "subgraph" in node_data:
        attrs["fx.subgraph"] = ir.StringAttr.get(str(node_data["subgraph"]))

    # Map Python-specific attributes
    for py_attr in (
        "vector_shapes",
        "reduction_dim",
        "iter_idx",
        "location",
        "expanded_dims",
        "scheduling_parameters",
    ):
        if py_attr in node_data:
            attrs[f"py.{py_attr}"] = ir.StringAttr.get(str(node_data[py_attr]))

    # Handle index attribute - store as string for now
    if "index" in node_data:
        index_str = node_data["index"]
        attrs["py.index"] = ir.StringAttr.get(str(index_str))

    return attrs


def create_unregistered_operation(op_name, node_data, insertion_point):
    """Fallback to create unregistered operation for unknown ops."""
    attrs = create_operation_attributes(node_data)

    return ir.Operation.create(
        name=op_name,
        attributes=attrs,
        results=[],
        operands=[],
    )


def main():
    try:
        data = json.load(sys.stdin)
    except Exception as e:
        print(f"FATAL: failed to read JSON: {e}", file=sys.stderr)
        return 3

    with ir.Context() as ctx:
        # Only allow registered dialects/ops
        try:
            ctx.allow_unregistered_dialects = False
        except Exception:
            pass
        wave.register_dialect(ctx)
        with ir.Location.unknown():
            module = ir.Module.create()
            # Build a function with arguments matching placeholders and optional returns.
            id_to_value = {}
            name_to_value = {}
            func_spec = data.get("function", {}) or {}
            func_args = func_spec.get("args", [])
            func_rets = func_spec.get("returns", [])
            # Derive argument types from corresponding placeholder nodes' wave_type if present.
            # Index func_args by id to find matching node metadata.
            nodes_by_id = {n.get("id"): n for n in data.get("nodes", [])}
            arg_types = []
            for arg in func_args:
                nid = arg.get("id")
                nmeta = nodes_by_id.get(nid, {}) if nid is not None else {}
                wave_type = nmeta.get("wave_type")
                if wave_type:
                    try:
                        arg_types.append(_type_from_wave_type_dict(ctx, wave_type))
                        continue
                    except Exception:
                        pass
                arg_types.append(_default_memory_tensor_type())
            ret_types = [_default_tensor_type() for _ in func_rets]
            func_type = ir.FunctionType.get(arg_types, ret_types)
            with ir.InsertionPoint(module.body):
                func_op = ir.Operation.create(
                    name="func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get("kernel"),
                        "function_type": ir.TypeAttr.get(func_type),
                    },
                    results=[],
                    operands=[],
                    regions=1,
                )
                entry_block = ir.Block.create_at_start(func_op.regions[0], arg_types)
                # Map function arguments by declared ids/names.
                for i, arg in enumerate(func_args):
                    arg_id = arg.get("id")
                    arg_name = arg.get("name", "")
                    if arg_id is not None:
                        id_to_value[arg_id] = entry_block.arguments[i]
                    if arg_name:
                        name_to_value[arg_name] = entry_block.arguments[i]

                # Emit body operations.
                with ir.InsertionPoint(entry_block):
                    for node in data.get("nodes", []):
                        tkw_op_name = node.get("tkw_op_name", "unknown")
                        # Skip ops that will not be materialized
                        if tkw_op_name in ("placeholder", "output"):
                            continue

                        # Try to create the operation using actual constructors or fallback
                        try:
                            # Prefer structured inputs (inputs_ex) falling back to names.
                            operands = []
                            if "inputs_ex" in node:
                                for ref in node.get("inputs_ex", []):
                                    pid = ref.get("id")
                                    ridx = int(ref.get("result", 0) or 0)
                                    val = id_to_value.get(pid)
                                    if val is not None:
                                        operands.append(val if ridx == 0 else None)
                            else:
                                for on in node.get("inputs", []):
                                    v = name_to_value.get(on)
                                    if v is not None:
                                        operands.append(v)

                            # Determine result type for ops that produce a single result.
                            result_type = None
                            if tkw_op_name in {
                                "add",
                                "mul",
                                "div",
                                "exp2",
                                "mma",
                                "read",
                                "register",
                            }:
                                wave_type = node.get("wave_type")
                                if wave_type:
                                    try:
                                        result_type = _type_from_wave_type_dict(
                                            ctx, wave_type
                                        )
                                    except Exception:
                                        # If wave_type parsing fails, we should have inferred the type
                                        # in mlir_converter.py, so this shouldn't happen
                                        result_type = None
                                else:
                                    # No wave_type available, we should have inferred the type
                                    # in mlir_converter.py, so this shouldn't happen
                                    result_type = None

                            attrs = create_operation_attributes(node)

                            # Try to use actual op constructor first
                            op = None
                            if tkw_op_name in WAVE_OP_CONSTRUCTORS:
                                ctor = WAVE_OP_CONSTRUCTORS[tkw_op_name]
                                # TODO: This could be more elegant
                                if (
                                    tkw_op_name in {"add", "mul", "div"}
                                    and len(operands) == 2
                                    and result_type is not None
                                ):
                                    op = ctor(result_type, operands[0], operands[1])
                                elif (
                                    tkw_op_name == "exp2"
                                    and len(operands) == 1
                                    and result_type is not None
                                ):
                                    op = ctor(result_type, operands[0])
                                elif (
                                    tkw_op_name == "read"
                                    and len(operands) == 1
                                    and result_type is not None
                                ):
                                    op = ctor(result_type, operands[0])
                                elif tkw_op_name == "write" and len(operands) == 2:
                                    op = ctor(operands[0], operands[1])
                                elif tkw_op_name == "register":
                                    if result_type is not None:
                                        # wave.register requires a scalar initialization value
                                        # Try to get the constant value from the trace, fallback to 0.0
                                        constant_value = 0.0
                                        if (
                                            "arg_2" in node
                                        ):  # The value is the 3rd argument (index 2) in NewRegister
                                            try:
                                                constant_value = float(node["arg_2"])
                                            except (ValueError, TypeError):
                                                print(
                                                    f"WARNING: Could not parse constant value '{node.get('arg_2', '')}', using 0.0",
                                                    file=sys.stderr,
                                                )

                                        try:
                                            # Try to get element type from wave tensor type
                                            element_type = result_type.element_type
                                        except AttributeError:
                                            # Fallback to f16 if we can't determine element type (matches test)
                                            element_type = ir.F16Type.get()

                                        # Create arith.constant for the scalar value
                                        constant_op = ir.Operation.create(
                                            name="arith.constant",
                                            attributes={
                                                "value": ir.FloatAttr.get(
                                                    element_type, constant_value
                                                )
                                            },
                                            results=[element_type],
                                            operands=[],
                                        )

                                        # Create the register op with the constant as operand
                                        op = ctor(result_type, constant_op.results[0])
                                    else:
                                        # Type inference failed, skip this operation
                                        print(
                                            f"WARNING: Could not infer type for register operation, skipping",
                                            file=sys.stderr,
                                        )
                                        continue

                            # Fallback to generic operation creation for unknown ops or when constructor fails
                            if op is None:
                                op_name = f"wave.{tkw_op_name if tkw_op_name != 'yield' else 'yield'}"
                                result_types = (
                                    [result_type] if result_type is not None else []
                                )
                                op = ir.Operation.create(
                                    name=op_name,
                                    attributes=attrs,
                                    results=result_types,
                                    operands=operands,
                                )

                            res = op.results[0] if len(op.results) > 0 else None
                            # Track by name and by optional node id.
                            if res is not None:
                                name_to_value[node.get("name", "")] = res
                                nid = node.get("id")
                                if nid is not None:
                                    id_to_value[nid] = res

                        except Exception as e:
                            print(
                                f"ERROR: failed to create op '{tkw_op_name}': {e}",
                                file=sys.stderr,
                            )
                    # Emit function return.
                    ret_operands = []
                    for r in func_rets:
                        pid = r.get("id")
                        ridx = int(r.get("result", 0) or 0)
                        v = id_to_value.get(pid)
                        if v is not None:
                            ret_operands.append(v if ridx == 0 else v)
                    # TODO: Import function from func dialect
                    ir.Operation.create(
                        name="func.return",
                        attributes={},
                        results=[],
                        operands=ret_operands,
                    )

            # Verify the module before printing
            module.operation.verify()
            print(module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
