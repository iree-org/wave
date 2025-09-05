# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import subprocess
import sys
from pathlib import Path
import os


def _dtype_str(dt):
    """Convert dtype to string representation."""
    try:
        return str(dt)
    except Exception:
        return "unknown"


def _addr_str(addr):
    """Convert address space to string representation."""
    from ..lang.kernel_buffer import AddressSpace

    if isinstance(addr, AddressSpace):
        return {
            AddressSpace.GLOBAL_MEMORY: "global",
            AddressSpace.SHARED_MEMORY: "shared",
            AddressSpace.REGISTER: "register",
        }[addr]
    # For symbolic address spaces, return as string (will be handled by water_emitter)
    return str(addr)


def _shape_syms(tt):
    """Extract symbolic shape from type."""
    syms = []
    for s in getattr(tt, "symbolic_shape", []) or []:
        try:
            syms.append(str(s))
        except Exception:
            syms.append("?")
    return syms


def _serialize_trace(
    trace, save_to_file: bool = False, filename: str | None = None
) -> dict:
    # Local import to avoid requiring package context for CLI mode.
    from ..ops.wave_ops import get_custom, Placeholder
    from ..lang.wave_types import Memory, Register
    from ..lang.kernel_buffer import AddressSpace
    import torch.fx as fx

    nodes = []
    walked = list(trace.walk())
    # Assign stable ids.
    name_to_id: dict[str, int] = {}
    for i, node in enumerate(walked):
        name_to_id[getattr(node, "name", f"n{i}")] = i

    for node in walked:
        custom = get_custom(node)

        # Call infer_type() on the custom object if it has the method
        if hasattr(custom, "infer_type"):
            try:
                custom.infer_type()
            except Exception as e:
                # If infer_type fails, continue without it
                pass

        entry = {
            "id": name_to_id.get(getattr(node, "name", ""), -1),
            "name": getattr(node, "name", ""),
            "op": getattr(node, "op", ""),
            "tkw_op_name": getattr(custom, "tkw_op_name", "unknown"),
            "subgraph": getattr(node, "subgraph_name", ""),
        }
        # Capture type info for placeholders.
        if isinstance(custom, Placeholder):
            t = getattr(custom, "_type", None)
            if t is not None:
                if issubclass(t, Memory):
                    shape_syms = _shape_syms(t)
                    addr = _addr_str(
                        getattr(t, "address_space", AddressSpace.GLOBAL_MEMORY)
                    )
                    dt = _dtype_str(getattr(t, "dtype", None))
                    entry["wave_type"] = {
                        "kind": "memory",
                        "shape": shape_syms,
                        "address_space": addr,
                        "dtype": dt,
                    }
                elif issubclass(t, Register):
                    shape_syms = _shape_syms(t)
                    dt = _dtype_str(getattr(t, "dtype", None))
                    entry["wave_type"] = {
                        "kind": "register",
                        "shape": shape_syms,
                        "address_space": "register",
                        "dtype": dt,
                    }
        # Capture result type for other ops when available on custom.
        t = getattr(custom, "type", None)
        if isinstance(t, type):
            if issubclass(t, Register):
                shape_syms = _shape_syms(t)
                dt = _dtype_str(getattr(t, "dtype", None))
                entry["wave_type"] = {
                    "kind": "register",
                    "shape": shape_syms,
                    "address_space": "register",
                    "dtype": dt,
                }
            elif issubclass(t, Memory):
                shape_syms = _shape_syms(t)
                dt = _dtype_str(getattr(t, "dtype", None))
                addr = getattr(t, "address_space", AddressSpace.GLOBAL_MEMORY)
                addr_str = _addr_str(addr)
                entry["wave_type"] = {
                    "kind": "memory",
                    "shape": shape_syms,
                    "address_space": addr_str,
                    "dtype": dt,
                }
        # Collect operand inputs by source node name (for wiring in emitter).
        inputs = []
        inputs_ex = []
        for arg in getattr(node, "args", ()):
            if isinstance(arg, fx.Node):
                inputs.append(getattr(arg, "name", ""))
                inputs_ex.append(
                    {"id": name_to_id.get(getattr(arg, "name", ""), -1), "result": 0}
                )
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, fx.Node):
                        inputs.append(getattr(a, "name", ""))
                        inputs_ex.append(
                            {
                                "id": name_to_id.get(getattr(a, "name", ""), -1),
                                "result": 0,
                            }
                        )
        if inputs:
            entry["inputs"] = inputs
            entry["inputs_ex"] = inputs_ex
        # Include selected python-side attributes as strings for MLIR attrs.
        for attr_name in (
            "index",
            "vector_shapes",
            "reduction_dim",
            "iter_idx",
            "location",
            "expanded_dims",
            "scheduling_parameters",
        ):
            if hasattr(node, attr_name):
                try:
                    entry[attr_name] = str(getattr(node, attr_name))
                except Exception:
                    entry[attr_name] = "<unserializable>"

        # Capture scalar arguments from custom ops (like value for register operations)
        if hasattr(node, "args") and node.args:
            for i, arg in enumerate(node.args):
                if not isinstance(arg, fx.Node) and not isinstance(arg, (list, tuple)):
                    # This is a scalar argument, capture it
                    entry[f"arg_{i}"] = str(arg)
        nodes.append(entry)
    # Top-level function IO description from placeholders/outputs.
    # Placeholders: ordered by arg_id if present; otherwise by appearance.
    placeholders = [n for n in walked if getattr(n, "op", "") == "placeholder"]

    def _arg_order(n):
        try:
            return int(getattr(n, "meta", {}).get("arg_id", 1 << 10))
        except Exception:
            return 1 << 10

    placeholders.sort(key=_arg_order)
    func_args = [
        {
            "id": name_to_id.get(getattr(n, "name", ""), -1),
            "name": getattr(n, "name", ""),
            "role": "input",
        }
        for n in placeholders
    ]
    # Returns: use first output node if present; collect its inputs.
    returns = []
    for n in walked:
        if getattr(n, "op", "") == "output":
            for arg in getattr(n, "args", ()):
                if isinstance(arg, fx.Node):
                    returns.append(
                        {
                            "id": name_to_id.get(getattr(arg, "name", ""), -1),
                            "result": 0,
                        }
                    )
            break
    payload = {"function": {"args": func_args, "returns": returns}, "nodes": nodes}
    if save_to_file or os.environ.get("WAVE_SAVE_TRACE", "0") in (
        "1",
        "true",
        "TRUE",
        "on",
        "ON",
    ):
        out_name = filename or "wave_trace.json"
        try:
            with open(out_name, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved trace JSON to {out_name}")
        except Exception as e:
            print(
                f"WARN: failed to save trace JSON to {out_name}: {e}", file=sys.stderr
            )
    return payload


def emit_wave_dialect(
    trace, save_trace: bool = False, trace_filename: str | None = None
):
    payload = _serialize_trace(trace, save_to_file=save_trace, filename=trace_filename)
    child = Path(__file__).with_name("water_emitter.py")
    if not child.exists():
        raise RuntimeError(f"water emitter helper not found: {child}")

    proc = subprocess.run(
        [sys.executable, str(child)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"water_mlir emitter failed (code {proc.returncode}):\n{proc.stderr}"
        )

    mlir_text = proc.stdout
    return mlir_text


if __name__ == "__main__":
    # Simple CLI to emit wave dialect from a JSON payload provided on stdin.
    # Expected format: {"nodes": [...]} matching _serialize_trace output.
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        print(f"FATAL: failed to read JSON: {e}", file=sys.stderr)
        sys.exit(3)

    child = Path(__file__).with_name("water_emitter.py")
    if not child.exists():
        print(f"FATAL: water emitter helper not found: {child}", file=sys.stderr)
        sys.exit(4)

    proc = subprocess.run(
        [sys.executable, str(child)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(
            f"water_mlir emitter failed (code {proc.returncode}):\n{proc.stderr}",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)

    sys.stdout.write(proc.stdout)
