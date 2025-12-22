# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Unified instruction emitter for AMDGCN kernels.

This module provides a single class that owns instruction definitions
and emission, supporting both:
- Direct assembly emission (legacy mode)
- Kernel IR emission (for whole-program register allocation)

The emitter generates methods dynamically from instruction definitions,
ensuring consistency between the instruction registry and emission API.

Usage:
    # Direct assembly emission
    emitter = UnifiedEmitter(architecture="gfx942", mode="direct")
    emitter.v_add_u32(dst="v0", src0="v1", src1="v2")
    print(emitter.get_lines())
    
    # Kernel IR emission
    from kernel_pipeline import KernelCompilationContext
    ctx = KernelCompilationContext()
    emitter = UnifiedEmitter(architecture="gfx942", mode="kernel_ir", context=ctx)
    result = emitter.v_add_u32(src0=v1, src1=v2)  # Returns virtual register
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import os

from .instruction_registry import (
    InstructionRegistry,
    InstructionDef,
    OperandType,
    InstructionCategory,
    get_registry,
)


# ==============================================================================
# Register Wrappers (to distinguish from immediates)
# ==============================================================================

class VReg:
    """Wrapper to explicitly mark a value as a VGPR index."""
    __slots__ = ('index',)
    
    def __init__(self, index: int):
        self.index = index
    
    def __repr__(self):
        return f"v{self.index}"


class SReg:
    """Wrapper to explicitly mark a value as an SGPR index."""
    __slots__ = ('index',)
    
    def __init__(self, index: int):
        self.index = index
    
    def __repr__(self):
        return f"s{self.index}"


class Imm:
    """Wrapper to explicitly mark a value as an immediate."""
    __slots__ = ('value',)
    
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
    DIRECT = auto()      # Emit directly to assembly lines
    KERNEL_IR = auto()   # Emit to KernelProgram via KernelCompilationContext


# ==============================================================================
# Operand Formatting
# ==============================================================================

def format_operand(value: Any, operand_types: Tuple[OperandType, ...], is_use: bool = True) -> str:
    """
    Format an operand value to its assembly string representation.
    
    Handles:
    - VReg/SReg/Imm wrappers for explicit typing
    - Physical register indices (int -> vN or sN)
    - Register strings (passthrough)
    - Immediate values (hex for large values)
    - Kernel IR register objects
    
    Args:
        value: The operand value
        operand_types: Allowed types for this operand
        is_use: True if this is a use (source), False if definition (dest)
    """
    if value is None:
        return ""
    
    # Handle explicit wrappers
    if isinstance(value, VReg):
        return f"v{value.index}"
    if isinstance(value, SReg):
        return f"s{value.index}"
    if isinstance(value, Imm):
        return repr(value)
    
    # Handle kernel IR register types (imported lazily to avoid circular deps)
    if hasattr(value, '__class__'):
        class_name = value.__class__.__name__
        if class_name in ('KVReg', 'KSReg', 'KPhysVReg', 'KPhysSReg', 'KSpecialReg', 'KRegRange'):
            # These will be resolved later by the kernel generator
            return str(value)
    
    # String operand (register name, label, etc.)
    if isinstance(value, str):
        return value
    
    # Tuple - register range (must check before int since (1,2) would iterate)
    if isinstance(value, tuple) and len(value) >= 2 and all(isinstance(v, int) for v in value):
        if OperandType.VGPR_PAIR in operand_types or OperandType.VGPR_QUAD in operand_types or OperandType.VGPR_16 in operand_types:
            return f"v[{value[0]}:{value[-1]}]"
        elif OperandType.SGPR_PAIR in operand_types or OperandType.SGPR_QUAD in operand_types:
            return f"s[{value[0]}:{value[-1]}]"
        # Default to VGPR for unknown tuple
        return f"v[{value[0]}:{value[-1]}]"
    
    # Integer - could be register index or immediate
    if isinstance(value, int):
        # For definitions, always treat as register index
        if not is_use:
            if OperandType.SGPR in operand_types or OperandType.SGPR_PAIR in operand_types:
                return f"s{value}"
            else:
                return f"v{value}"
        
        # For uses, check if immediate is allowed
        has_imm = OperandType.IMM in operand_types or OperandType.IMM16 in operand_types
        only_reg = not has_imm and (
            OperandType.VGPR in operand_types or 
            OperandType.SGPR in operand_types or
            OperandType.VGPR_PAIR in operand_types or
            OperandType.SGPR_PAIR in operand_types
        )
        
        if only_reg:
            # Must be a register
            if OperandType.SGPR in operand_types or OperandType.SGPR_PAIR in operand_types:
                return f"s{value}"
            else:
                return f"v{value}"
        else:
            # Can be immediate - format as immediate
            if abs(value) > 0xFFFF:
                return f"0x{value & 0xFFFFFFFF:x}"
            elif value < 0:
                return str(value)
            else:
                return str(value)
    
    return str(value)


def format_offset(value: int) -> str:
    """Format an offset value."""
    if value == 0:
        return ""
    return f"offset:{value}"


# ==============================================================================
# Instruction Builder
# ==============================================================================

@dataclass
class InstructionBuilder:
    """
    Builds assembly line for a single instruction.
    
    Uses instruction definition to properly format operands.
    """
    instr_def: InstructionDef
    
    def build(
        self,
        defs: List[Any] = None,
        uses: List[Any] = None,
        comment: str = None,
    ) -> str:
        """Build assembly line for the instruction."""
        defs = defs or []
        uses = uses or []
        
        # Handle pseudo-ops
        if self.instr_def.category == InstructionCategory.PSEUDO:
            return self._build_pseudo(defs, uses, comment)
        
        # Build operand strings
        operands = []
        
        # Add destinations (is_use=False)
        for i, def_op in enumerate(self.instr_def.defs):
            if i < len(defs):
                operands.append(format_operand(defs[i], def_op.types, is_use=False))
        
        # Add sources (is_use=True)
        for i, use_op in enumerate(self.instr_def.uses):
            if i < len(uses):
                value = uses[i]
                
                # Handle offset specially
                if OperandType.OFFSET in use_op.types:
                    if value and value != 0:
                        # Offset formatting depends on instruction
                        if self.instr_def.offset_format == "space_separated":
                            # Will be handled in post-processing
                            operands.append(f"offset:{value}")
                        else:
                            operands.append(f"offset:{value}")
                    # Skip if offset is 0 and optional
                    elif use_op.optional:
                        continue
                    else:
                        operands.append(format_operand(value, use_op.types, is_use=True))
                else:
                    operands.append(format_operand(value, use_op.types, is_use=True))
        
        # Build line
        mnemonic = self.instr_def.mnemonic
        
        if not operands:
            line = f"    {mnemonic}"
        else:
            line = f"    {mnemonic} {', '.join(operands)}"
        
        # Apply special formatting
        line = self._apply_special_formatting(line, operands)
        
        # Add LDS modifier if needed
        if self.instr_def.lds_modifier:
            if "  //" in line:
                parts = line.split("  //")
                line = parts[0] + " lds  //" + parts[1]
            else:
                line = line + " lds"
        
        # Add comment
        if comment:
            line += f"  // {comment}"
        
        return line
    
    def _build_pseudo(self, defs: List[Any], uses: List[Any], comment: str) -> str:
        """Build pseudo-op line."""
        if self.instr_def.name == "_comment":
            return f"    // {comment or uses[0] if uses else ''}"
        elif self.instr_def.name == "_label":
            label = uses[0] if uses else comment
            return f"{label}:"
        elif self.instr_def.name == "_raw_asm":
            return uses[0] if uses else comment
        return ""
    
    def _apply_special_formatting(self, line: str, operands: List[str]) -> str:
        """Apply instruction-specific formatting rules."""
        # DS instructions: offset is space-separated
        if self.instr_def.offset_format == "space_separated":
            if ", offset:" in line:
                parts = line.split(", offset:")
                if len(parts) == 2:
                    line = parts[0] + " offset:" + parts[1]
        
        # Buffer instructions: special formatting
        # For loads: mnemonic dst, vaddr, srd, soffset offen [offset:N]
        # For stores: mnemonic src, vaddr, srd, soffset offen [offset:N]
        # For lds loads: mnemonic vaddr, srd, soffset offen [offset:N] lds
        if self.instr_def.mnemonic.startswith("buffer_"):
            # Split into mnemonic and operands
            if " " in line:
                prefix, rest = line.split(" ", 1)
                parts = rest.split(", ")
                
                # Find where comma-separated operands end and modifiers begin
                # LDS loads have 3 operands, regular have 4
                is_lds = self.instr_def.lds_modifier
                min_parts = 3 if is_lds else 4
                
                if len(parts) >= min_parts:
                    # First min_parts comma-separated, then add offen and offset
                    base = ", ".join(parts[:min_parts])
                    modifiers = ["offen"]
                    for part in parts[min_parts:]:
                        if part.startswith("offset:"):
                            modifiers.append(part)
                    line = f"{prefix} {base} {' '.join(modifiers)}"
        
        # Branch instructions: label from comment
        if self.instr_def.is_branch and not operands:
            # Label might be in operands or needs special handling
            pass
        
        return line


# ==============================================================================
# Unified Emitter
# ==============================================================================

class UnifiedEmitter:
    """
    Unified instruction emitter supporting both direct and kernel IR emission.
    
    This class dynamically generates emission methods from the instruction
    registry, providing a consistent API regardless of emission mode.
    
    In DIRECT mode:
        - Methods append assembly lines to internal buffer
        - Caller retrieves lines with get_lines()
    
    In KERNEL_IR mode:
        - Methods emit to a KernelCompilationContext
        - Methods return virtual register results
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
    
    def _emit_direct(
        self,
        instr: InstructionDef,
        args: tuple,
        kwargs: dict,
        comment: str,
    ) -> None:
        """Emit instruction directly to assembly lines."""
        # Parse positional args into defs and uses
        defs = []
        uses = []
        
        arg_idx = 0
        for def_op in instr.defs:
            if arg_idx < len(args):
                defs.append(args[arg_idx])
                arg_idx += 1
            elif def_op.name in kwargs:
                defs.append(kwargs[def_op.name])
        
        for use_op in instr.uses:
            if arg_idx < len(args):
                uses.append(args[arg_idx])
                arg_idx += 1
            elif use_op.name in kwargs:
                uses.append(kwargs[use_op.name])
            elif use_op.optional:
                uses.append(None)
        
        # Build and emit
        builder = InstructionBuilder(instr)
        line = builder.build(defs, uses, comment)
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
        
        # Fallback: emit raw if method not available
        builder = InstructionBuilder(instr)
        
        # Parse args
        defs = []
        uses = []
        arg_idx = 0
        for def_op in instr.defs:
            if arg_idx < len(args):
                defs.append(args[arg_idx])
                arg_idx += 1
            elif def_op.name in kwargs:
                defs.append(kwargs[def_op.name])
        
        for use_op in instr.uses:
            if arg_idx < len(args):
                uses.append(args[arg_idx])
                arg_idx += 1
            elif use_op.name in kwargs:
                uses.append(kwargs[use_op.name])
            elif use_op.optional:
                uses.append(None)
        
        line = builder.build(defs, uses, comment)
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
        self._lines.append(f"    // {text}")
    
    def emit_label(self, name: str) -> None:
        """Emit a label."""
        self._lines.append(f"{name}:")
    
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

