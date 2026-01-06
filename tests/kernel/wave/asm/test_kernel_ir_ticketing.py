# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from wave_lang.kernel.wave.asm.kernel_compilation_context import (
    KernelCompilationContext,
)
from wave_lang.kernel.wave.asm.kernel_ir import KInstr, KVReg, KRegRange, KImm


def test_ticketing_inserts_waitcnt_before_consumer(monkeypatch):
    # Enable the pass for this test only.
    monkeypatch.setenv("WAVE_KERNEL_TICKETING", "1")

    ctx = KernelCompilationContext(
        use_flat_tid=False, use_workgroup_ids=(False, False, False)
    )

    # Fake a VMEM load defining v0..v3.
    load_dst = KRegRange(KVReg(0), 4, alignment=4)
    ctx.program.emit(
        KInstr(
            "buffer_load_dwordx4",
            defs=(load_dst,),
            uses=(KVReg(10),),
            comment="vmem load",
        )
    )

    # Consumer uses one component reg (v1).
    ctx.program.emit(
        KInstr(
            "v_add_u32",
            defs=(KVReg(20),),
            uses=(KVReg(1), KImm(1)),
            comment="consume load",
        )
    )

    # Apply ticketing wait placement directly (avoid regalloc/asm generation).
    ctx._apply_ticketing_waitcnt_placement()

    # Expect a waitcnt inserted before v_add_u32.
    names = [i.name for i in ctx.program.instructions]
    assert names == ["buffer_load_dwordx4", "s_waitcnt", "v_add_u32"]
    assert ctx.program.instructions[1].uses == ("vmcnt(0)",)


def test_ticketing_coalesces_waits(monkeypatch):
    monkeypatch.setenv("WAVE_KERNEL_TICKETING", "1")
    ctx = KernelCompilationContext(
        use_flat_tid=False, use_workgroup_ids=(False, False, False)
    )

    # Two VMEM loads.
    dst0 = KRegRange(KVReg(0), 4, alignment=4)
    dst1 = KRegRange(KVReg(4), 4, alignment=4)
    ctx.program.emit(
        KInstr(
            "buffer_load_dwordx4", defs=(dst0,), uses=(KVReg(10),), comment="vmem load0"
        )
    )
    ctx.program.emit(
        KInstr(
            "buffer_load_dwordx4", defs=(dst1,), uses=(KVReg(11),), comment="vmem load1"
        )
    )

    # Two consumers; ticketing should avoid inserting redundant waits for the second consumer.
    ctx.program.emit(
        KInstr(
            "v_add_u32",
            defs=(KVReg(20),),
            uses=(KVReg(1), KImm(1)),
            comment="consume load0",
        )
    )
    ctx.program.emit(
        KInstr(
            "v_add_u32",
            defs=(KVReg(21),),
            uses=(KVReg(5), KImm(1)),
            comment="consume load1",
        )
    )

    ctx._apply_ticketing_waitcnt_placement()

    names = [i.name for i in ctx.program.instructions]
    assert names == [
        "buffer_load_dwordx4",
        "buffer_load_dwordx4",
        "s_waitcnt",
        "v_add_u32",
        "s_waitcnt",
        "v_add_u32",
    ]
    assert ctx.program.instructions[2].uses == ("vmcnt(1)",)
    assert ctx.program.instructions[4].uses == ("vmcnt(0)",)
