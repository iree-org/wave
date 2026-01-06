# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Unified instruction emitter for AMDGCN kernels.

This module provides a single class that owns instruction definitions
and emission, supporting both:
- Direct assembly emission (legacy mode - deprecated)
- Kernel IR emission (for whole-program register allocation)

The emitter generates methods dynamically from instruction definitions,
ensuring consistency between the instruction registry and emission API.

All instruction formatting goes through InstructionFormatter, which is
the SINGLE point for physical instruction rendering.

Usage:
    # Direct assembly emission (deprecated - use kernel IR instead)
    emitter = UnifiedEmitter(architecture="gfx942", mode="direct")
    emitter.v_add_u32(dst="v0", src0="v1", src1="v2")
    print(emitter.get_lines())

    # Kernel IR emission (preferred)
    from kernel_compilation_context import KernelCompilationContext
    ctx = KernelCompilationContext()
    emitter = UnifiedEmitter(architecture="gfx942", mode="kernel_ir", context=ctx)
    result = emitter.v_add_u32(src0=v1, src1=v2)  # Returns virtual register
"""

from enum import Enum, auto
from typing import List, Optional, Any, Callable

from .instruction_registry import (
    InstructionRegistry,
    InstructionDef,
    get_registry,
)
from .instruction_formatter import InstructionFormatter, get_formatter


# ==============================================================================
# Register Wrappers (to distinguish from immediates)
# ==============================================================================


class VReg:
    """Wrapper to explicitly mark a value as a VGPR index."""

    __slots__ = ("index",)

    def __init__(self, index: int):
        self.index = index

    def __repr__(self):
        return f"v{self.index}"


class SReg:
    """Wrapper to explicitly mark a value as an SGPR index."""

    __slots__ = ("index",)

    def __init__(self, index: int):
        self.index = index

    def __repr__(self):
        return f"s{self.index}"


class Imm:
    """Wrapper to explicitly mark a value as an immediate."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def __repr__(self):
        if abs(self.value) > 0xFFFF:
            return f"0x{self.value & 0xFFFFFFFF:x}"
        return str(self.value)


# ==============================================================================
# Emission Modes
# ==============================================================================


class EmissionMode(Enum):
    """Mode of instruction emission."""

    DIRECT = auto()  # Emit directly to assembly lines (deprecated)
    KERNEL_IR = auto()  # Emit to KernelProgram via KernelCompilationContext


# ==============================================================================
# Unified Emitter
# ==============================================================================


class UnifiedEmitter:
    """
    Unified instruction emitter supporting both direct and kernel IR emission.

    This class dynamically generates emission methods from the instruction
    registry, providing a consistent API regardless of emission mode.

    In DIRECT mode (deprecated):
        - Methods append assembly lines to internal buffer
        - Caller retrieves lines with get_lines()

    In KERNEL_IR mode (preferred):
        - Methods emit to a KernelCompilationContext
        - Methods return virtual register results

    All physical instruction formatting goes through InstructionFormatter.
    """

    def __init__(
        self,
        architecture: str = "common",
        mode: EmissionMode = EmissionMode.DIRECT,
        context: Any = None,  # KernelCompilationContext for KERNEL_IR mode
    ):
        self.architecture = architecture
        self.mode = mode
        self.context = context
        self._registry = get_registry(architecture)
        self._formatter = get_formatter(architecture)
        self._lines: List[str] = []

        # Generate emission methods
        self._generate_methods()

    def _generate_methods(self) -> None:
        """Generate emission methods for all instructions in the registry."""
        for instr in self._registry:
            if not instr.name.startswith("_"):  # Skip pseudo-ops for direct methods
                method = self._create_emission_method(instr)
                setattr(self, instr.name, method)

    def _create_emission_method(self, instr: InstructionDef) -> Callable:
        """Create an emission method for an instruction."""

        def emit_method(*args, comment: str = None, **kwargs):
            """Emit instruction."""
            if self.mode == EmissionMode.DIRECT:
                return self._emit_direct(instr, args, kwargs, comment)
            else:
                return self._emit_kernel_ir(instr, args, kwargs, comment)

        # Set docstring
        emit_method.__doc__ = f"""
        Emit {instr.mnemonic} instruction.

        Mnemonic: {instr.mnemonic}
        Category: {instr.category.name}
        Latency: {instr.latency}
        """

        return emit_method

    def _parse_args(self, instr: InstructionDef, args: tuple, kwargs: dict):
        """Parse positional and keyword args into defs and uses."""
        defs = []
        uses = []

        arg_idx = 0
        for def_op in instr.defs:
            if arg_idx < len(args):
                defs.append(self._format_arg(args[arg_idx]))
                arg_idx += 1
            elif def_op.name in kwargs:
                defs.append(self._format_arg(kwargs[def_op.name]))

        for use_op in instr.uses:
            if arg_idx < len(args):
                uses.append(self._format_arg(args[arg_idx]))
                arg_idx += 1
            elif use_op.name in kwargs:
                uses.append(self._format_arg(kwargs[use_op.name]))
            elif use_op.optional:
                uses.append(None)

        return defs, uses

    def _format_arg(self, value: Any) -> Any:
        """Format an argument value, handling wrappers."""
        if isinstance(value, VReg):
            return f"v{value.index}"
        if isinstance(value, SReg):
            return f"s{value.index}"
        if isinstance(value, Imm):
            return repr(value)
        return value

    def _emit_direct(
        self,
        instr: InstructionDef,
        args: tuple,
        kwargs: dict,
        comment: str,
    ) -> None:
        """Emit instruction directly to assembly lines using InstructionFormatter."""
        defs, uses = self._parse_args(instr, args, kwargs)

        # Use InstructionFormatter for all formatting
        line = self._formatter.format(instr.name, defs=defs, uses=uses, comment=comment)
        if line:
            self._lines.append(line)

    def _emit_kernel_ir(
        self,
        instr: InstructionDef,
        args: tuple,
        kwargs: dict,
        comment: str,
    ) -> Any:
        """Emit instruction to kernel IR via context."""
        if self.context is None:
            raise ValueError("KernelCompilationContext required for KERNEL_IR mode")

        # Map instruction to context method if available
        method_name = instr.name
        if hasattr(self.context, method_name):
            ctx_method = getattr(self.context, method_name)
            # Call context method with provided args
            return ctx_method(*args, comment=comment, **kwargs)

        # Fallback: emit raw using InstructionFormatter
        defs, uses = self._parse_args(instr, args, kwargs)
        line = self._formatter.format(instr.name, defs=defs, uses=uses, comment=comment)
        if line:
            self.context.emit_raw(line)

        return None

    # =========================================================================
    # Line Management
    # =========================================================================

    def get_lines(self) -> List[str]:
        """Get all emitted assembly lines."""
        return self._lines.copy()

    def clear(self) -> None:
        """Clear emitted lines."""
        self._lines.clear()

    def emit_line(self, line: str) -> None:
        """Emit a raw line (for custom formatting)."""
        self._lines.append(line)

    def emit_comment(self, text: str) -> None:
        """Emit a comment line."""
        self._lines.append(self._formatter.format_comment(text))

    def emit_label(self, name: str) -> None:
        """Emit a label."""
        self._lines.append(self._formatter.format_label(name))

    def emit_blank(self) -> None:
        """Emit a blank line."""
        self._lines.append("")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_instruction_def(self, name: str) -> Optional[InstructionDef]:
        """Get instruction definition by name."""
        return self._registry.get(name)

    def get_latency(self, name: str) -> int:
        """Get instruction latency."""
        instr = self._registry.get(name)
        return instr.latency if instr else 1

    @property
    def registry(self) -> InstructionRegistry:
        """Get the underlying instruction registry."""
        return self._registry

    @property
    def formatter(self) -> InstructionFormatter:
        """Get the underlying instruction formatter."""
        return self._formatter


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_direct_emitter(architecture: str = "common") -> UnifiedEmitter:
    """Create an emitter for direct assembly emission."""
    return UnifiedEmitter(architecture=architecture, mode=EmissionMode.DIRECT)


def create_kernel_ir_emitter(
    context: Any,
    architecture: str = "common",
) -> UnifiedEmitter:
    """Create an emitter for kernel IR emission."""
    return UnifiedEmitter(
        architecture=architecture,
        mode=EmissionMode.KERNEL_IR,
        context=context,
    )
