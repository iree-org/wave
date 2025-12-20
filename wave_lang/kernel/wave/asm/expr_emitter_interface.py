# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Expression emitter interface protocol.

This module defines the contract that all expression emitters must implement.
Both the legacy ExprEmitter and the new ExprEmitterV2 conform to this interface.

Call sites:
    - handlers.py:_get_expr_emitter() - Creates and caches emitters per kernel
    - handlers.py - Calls get_or_emit() for address calculations
    - gather_to_shared.py - Calls get_or_emit() for LDS address calculations
    - utils.py:emit_expression_asm() - Legacy path that creates fresh emitters

The standard flow is:
    1. Create emitter: emitter = ExprEmitter(asm_emitter, kernel_info)
    2. Bind symbols: emitter.bind_symbol("wgid_x", "s2")
    3. Emit expressions: reg = emitter.get_or_emit(sympy_expr)
    4. Clear cache (optional): emitter.clear_cache()
"""

import os
import sympy
from typing import Optional, Protocol, Dict, runtime_checkable


# Environment variable to select emitter implementation
# "v2" (default): Use ExprEmitterV2 with virtual registers
# "legacy": Use original ExprEmitter
EXPR_EMITTER_ENV = "WAVE_EXPR_EMITTER"
EXPR_EMITTER_DEFAULT = "v2"


def get_expr_emitter_version() -> str:
    """Get the configured expression emitter version."""
    return os.environ.get(EXPR_EMITTER_ENV, EXPR_EMITTER_DEFAULT)


def use_v2_emitter() -> bool:
    """Check if v2 emitter should be used."""
    return get_expr_emitter_version() != "legacy"


@runtime_checkable
class ExprEmitterProtocol(Protocol):
    """
    Protocol defining the expression emitter interface.
    
    All expression emitters (ExprEmitter, ExprEmitterV2) must implement
    these methods to be usable by the ASM backend.
    """
    
    def bind_symbol(self, symbol_name: str, register: str) -> None:
        """
        Bind a symbol name to a register.
        
        Called during emitter setup to map symbolic names to physical registers.
        
        Args:
            symbol_name: Symbol name (e.g., "wgid_x", "tid_x")
            register: Register name (e.g., "s2", "v0")
        """
        ...
    
    def get_or_emit(
        self, 
        expr: sympy.Expr, 
        dst_hint: Optional[str] = None
    ) -> str:
        """
        Get cached register for expression or emit and cache it.
        
        This is the primary entry point for expression emission. It:
        1. Checks if the expression has already been computed (CSE)
        2. If cached, returns the cached register
        3. If not cached, emits instructions and caches the result
        
        Args:
            expr: SymPy expression to emit
            dst_hint: Optional destination register hint (e.g., "v2")
                     The actual register may differ if CSE hits or
                     fused instruction patterns are used.
        
        Returns:
            Register string (e.g., "v5") containing the expression result.
        """
        ...
    
    def emit(self, expr: sympy.Expr, dst_reg: str) -> str:
        """
        Emit instructions for an expression into a specific register.
        
        Lower-level than get_or_emit - always emits, no caching.
        Used by legacy code paths that need direct control.
        
        Args:
            expr: SymPy expression to emit
            dst_reg: Destination register (e.g., "v2")
        
        Returns:
            The register containing the result (usually dst_reg).
        """
        ...
    
    def clear_cache(self) -> None:
        """
        Clear the expression cache.
        
        Should be called at kernel boundaries to prevent stale CSE hits.
        """
        ...
    
    def maybe_dump_summary(self) -> None:
        """
        Dump CSE summary if enabled via environment variable.
        
        Set WAVE_EXPR_V2_CSE_SUMMARY=1 to enable.
        """
        ...


def create_expr_emitter(asm_emitter, kernel_info) -> ExprEmitterProtocol:
    """
    Factory function to create an expression emitter.
    
    Selects between ExprEmitterV2 (default) and legacy ExprEmitter
    based on the WAVE_EXPR_EMITTER environment variable.
    
    Args:
        asm_emitter: The AsmEmitter instance
        kernel_info: Kernel configuration info
        
    Returns:
        An expression emitter implementing ExprEmitterProtocol
    """
    if use_v2_emitter():
        from .expr_emitter_v2 import ExprEmitterV2
        return ExprEmitterV2(asm_emitter, kernel_info)
    else:
        from .expression_emitter import ExprEmitter
        return ExprEmitter(asm_emitter, kernel_info)

