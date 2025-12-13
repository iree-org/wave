# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for VMEM and LGKM ticketing system.
"""

import pytest
from wave_lang.kernel.wave.asm.ticketing import Ticketing


class TestVMEMTicketing:
    """Tests for VMEM (vector memory) ticketing."""

    def test_vmem_ticket_initialization(self):
        """Test initial state of VMEM ticketing."""
        ticketing = Ticketing()

        assert ticketing._vmem_last_ticket == -1
        assert ticketing._vmem_last_wait_threshold is None

    def test_vmem_ticket_issuance(self):
        """Test issuing VMEM tickets."""
        ticketing = Ticketing()

        t0 = ticketing.next_vmem_ticket()
        t1 = ticketing.next_vmem_ticket()
        t2 = ticketing.next_vmem_ticket()

        assert t0 == 0
        assert t1 == 1
        assert t2 == 2
        assert ticketing._vmem_last_ticket == 2

    def test_vmem_compute_wait_first_time(self):
        """Test computing VMEM wait for the first time."""
        ticketing = Ticketing()

        # Issue 3 VMEM operations
        ticketing.next_vmem_ticket()  # ticket 0
        ticketing.next_vmem_ticket()  # ticket 1
        ticketing.next_vmem_ticket()  # ticket 2

        # Wait for ticket 0: allow tickets 1 and 2 to remain in-flight
        threshold = ticketing.compute_vmem_wait(0)
        assert threshold == 2  # max(0, 2 - 0) = 2
        assert ticketing._vmem_last_wait_threshold == 2

    def test_vmem_compute_wait_coalescing(self):
        """Test VMEM wait coalescing (returns None for less strict waits)."""
        ticketing = Ticketing()

        # Issue 5 VMEM operations (tickets 0-4)
        for _ in range(5):
            ticketing.next_vmem_ticket()

        # First wait for ticket 1: threshold = 4-1 = 3
        threshold1 = ticketing.compute_vmem_wait(1)
        assert threshold1 == 3

        # Try to wait for ticket 0 (less strict): should return None
        threshold2 = ticketing.compute_vmem_wait(0)
        assert threshold2 is None  # Already waiting for 3, which is stricter

        # Wait for ticket 3 (more strict): threshold = 4-3 = 1
        threshold3 = ticketing.compute_vmem_wait(3)
        assert threshold3 == 1  # More strict, so emitted

    def test_vmem_ticket_invalidates_wait_threshold(self):
        """Test that issuing a new ticket invalidates the wait threshold."""
        ticketing = Ticketing()

        # Issue and wait
        ticketing.next_vmem_ticket()
        ticketing.compute_vmem_wait(0)
        assert ticketing._vmem_last_wait_threshold == 0

        # Issue new ticket - should clear wait threshold
        ticketing.next_vmem_ticket()
        assert ticketing._vmem_last_wait_threshold is None

    def test_vmem_zero_threshold(self):
        """Test VMEM wait with zero threshold (drain all)."""
        ticketing = Ticketing()

        # Issue tickets
        ticketing.next_vmem_ticket()
        ticketing.next_vmem_ticket()

        # Wait for last ticket: threshold should be 0
        threshold = ticketing.compute_vmem_wait(1)
        assert threshold == 0  # max(0, 1-1) = 0


class TestLGKMTicketing:
    """Tests for LGKM (LDS/scalar memory) ticketing."""

    def test_lgkm_ticket_initialization(self):
        """Test initial state of LGKM ticketing."""
        ticketing = Ticketing()

        assert ticketing._lgkm_last_ticket == -1
        assert ticketing._lgkm_last_wait_threshold is None

    def test_lgkm_ticket_issuance(self):
        """Test issuing LGKM tickets."""
        ticketing = Ticketing()

        t0 = ticketing.next_lgkm_ticket()
        t1 = ticketing.next_lgkm_ticket()
        t2 = ticketing.next_lgkm_ticket()

        assert t0 == 0
        assert t1 == 1
        assert t2 == 2
        assert ticketing._lgkm_last_ticket == 2

    def test_lgkm_compute_wait_first_time(self):
        """Test computing LGKM wait for the first time."""
        ticketing = Ticketing()

        # Issue 3 LGKM operations
        ticketing.next_lgkm_ticket()  # ticket 0
        ticketing.next_lgkm_ticket()  # ticket 1
        ticketing.next_lgkm_ticket()  # ticket 2

        # Wait for ticket 0: allow tickets 1 and 2 to remain in-flight
        threshold = ticketing.compute_lgkm_wait(0)
        assert threshold == 2  # max(0, 2 - 0) = 2
        assert ticketing._lgkm_last_wait_threshold == 2

    def test_lgkm_compute_wait_coalescing(self):
        """Test LGKM wait coalescing (returns None for less strict waits)."""
        ticketing = Ticketing()

        # Issue 4 LGKM operations
        for _ in range(4):
            ticketing.next_lgkm_ticket()

        # First wait for ticket 1: threshold = 3-1 = 2
        threshold1 = ticketing.compute_lgkm_wait(1)
        assert threshold1 == 2

        # Try to wait for ticket 0 (less strict): should return None
        threshold2 = ticketing.compute_lgkm_wait(0)
        assert threshold2 is None  # Already waiting for 2, which is stricter

        # Wait for ticket 2 (more strict): threshold = 3-2 = 1
        threshold3 = ticketing.compute_lgkm_wait(2)
        assert threshold3 == 1  # More strict, so emitted

    def test_lgkm_ticket_invalidates_wait_threshold(self):
        """Test that issuing a new ticket invalidates the wait threshold."""
        ticketing = Ticketing()

        # Issue and wait
        ticketing.next_lgkm_ticket()
        ticketing.compute_lgkm_wait(0)
        assert ticketing._lgkm_last_wait_threshold == 0

        # Issue new ticket - should clear wait threshold
        ticketing.next_lgkm_ticket()
        assert ticketing._lgkm_last_wait_threshold is None

    def test_lgkm_observe_wait(self):
        """Test observing external LGKM waits."""
        ticketing = Ticketing()

        # No wait observed yet
        assert ticketing._lgkm_last_wait_threshold is None

        # Observe a wait to threshold 2
        ticketing.observe_lgkm_wait(2)
        assert ticketing._lgkm_last_wait_threshold == 2

        # Observe a stricter wait (lower threshold)
        ticketing.observe_lgkm_wait(1)
        assert ticketing._lgkm_last_wait_threshold == 1

        # Observe a less strict wait (higher threshold) - should be ignored
        ticketing.observe_lgkm_wait(3)
        assert ticketing._lgkm_last_wait_threshold == 1  # Unchanged


class TestLGKMTicketingWithBarriers:
    """Tests for LGKM ticketing with user-specified barriers."""

    def test_barrier_with_outstanding_lgkm(self):
        """Test barrier emits lgkmcnt(0) when LGKM ops are outstanding."""
        ticketing = Ticketing()

        # Issue some LGKM operations
        t0 = ticketing.next_lgkm_ticket()
        t1 = ticketing.next_lgkm_ticket()

        assert t0 == 0
        assert t1 == 1
        assert ticketing._lgkm_last_ticket == 1
        assert ticketing._lgkm_last_wait_threshold is None

        # Simulate barrier: check if wait is needed
        has_outstanding = (
            ticketing._lgkm_last_ticket >= 0
            and ticketing._lgkm_last_wait_threshold != 0
        )
        assert has_outstanding is True  # Should emit lgkmcnt(0)

        # Observe the wait
        ticketing.observe_lgkm_wait(0)
        assert ticketing._lgkm_last_wait_threshold == 0

    def test_barrier_without_outstanding_lgkm(self):
        """Test barrier skips lgkmcnt(0) when no LGKM ops are outstanding."""
        ticketing = Ticketing()

        # No LGKM operations issued
        assert ticketing._lgkm_last_ticket == -1

        # Simulate barrier: check if wait is needed
        has_outstanding = (
            ticketing._lgkm_last_ticket >= 0
            and ticketing._lgkm_last_wait_threshold != 0
        )
        assert has_outstanding is False  # Should NOT emit lgkmcnt(0)

        # Still observe the wait (barrier implies drain)
        ticketing.observe_lgkm_wait(0)
        assert ticketing._lgkm_last_wait_threshold == 0

    def test_consecutive_barriers_coalesce(self):
        """Test consecutive barriers emit only one lgkmcnt(0)."""
        ticketing = Ticketing()

        # Issue LGKM operation
        ticketing.next_lgkm_ticket()

        # First barrier: should emit lgkmcnt(0)
        has_outstanding_1 = (
            ticketing._lgkm_last_ticket >= 0
            and ticketing._lgkm_last_wait_threshold != 0
        )
        assert has_outstanding_1 is True
        ticketing.observe_lgkm_wait(0)

        # Second barrier immediately after: should NOT emit lgkmcnt(0)
        has_outstanding_2 = (
            ticketing._lgkm_last_ticket >= 0
            and ticketing._lgkm_last_wait_threshold != 0
        )
        assert has_outstanding_2 is False  # Already drained
        ticketing.observe_lgkm_wait(0)  # Still observe for consistency

    def test_barrier_then_new_lgkm_then_barrier(self):
        """Test barrier, new LGKM op, then another barrier."""
        ticketing = Ticketing()

        # Issue LGKM and barrier
        ticketing.next_lgkm_ticket()
        ticketing.observe_lgkm_wait(0)
        assert ticketing._lgkm_last_wait_threshold == 0

        # Issue new LGKM operation after barrier
        t2 = ticketing.next_lgkm_ticket()
        assert t2 == 1
        assert ticketing._lgkm_last_wait_threshold is None  # Reset by next_lgkm_ticket

        # Another barrier: should emit lgkmcnt(0) again for new LGKM op
        has_outstanding = (
            ticketing._lgkm_last_ticket >= 0
            and ticketing._lgkm_last_wait_threshold != 0
        )
        assert has_outstanding is True
        ticketing.observe_lgkm_wait(0)

    def test_compute_lgkm_wait_after_observe(self):
        """Test compute_lgkm_wait returns None after observe(0)."""
        ticketing = Ticketing()

        # Issue LGKM operations
        t0 = ticketing.next_lgkm_ticket()
        t1 = ticketing.next_lgkm_ticket()

        # Observe lgkmcnt(0) (as barrier would)
        ticketing.observe_lgkm_wait(0)

        # Try to compute wait for any ticket - should return None (already drained)
        wait = ticketing.compute_lgkm_wait(t0)
        assert wait is None

        wait = ticketing.compute_lgkm_wait(t1)
        assert wait is None

    def test_observe_does_not_reset_tickets(self):
        """Test observe_lgkm_wait does not reset ticket counter."""
        ticketing = Ticketing()

        # Issue LGKM operations
        ticketing.next_lgkm_ticket()  # ticket 0
        ticketing.next_lgkm_ticket()  # ticket 1

        # Observe wait
        ticketing.observe_lgkm_wait(0)

        # Ticket counter should remain unchanged
        assert ticketing._lgkm_last_ticket == 1

        # Next ticket should be 2, not 0
        t_next = ticketing.next_lgkm_ticket()
        assert t_next == 2


class TestTicketingIndependence:
    """Tests to ensure VMEM and LGKM ticketing are independent."""

    def test_vmem_and_lgkm_independent_tickets(self):
        """Test VMEM and LGKM ticket counters are independent."""
        ticketing = Ticketing()

        # Issue mixed operations
        v0 = ticketing.next_vmem_ticket()
        l0 = ticketing.next_lgkm_ticket()
        v1 = ticketing.next_vmem_ticket()
        l1 = ticketing.next_lgkm_ticket()

        # Each counter tracks independently
        assert v0 == 0 and v1 == 1
        assert l0 == 0 and l1 == 1
        assert ticketing._vmem_last_ticket == 1
        assert ticketing._lgkm_last_ticket == 1

    def test_vmem_wait_does_not_affect_lgkm(self):
        """Test VMEM wait computation doesn't affect LGKM."""
        ticketing = Ticketing()

        # Issue both types
        ticketing.next_vmem_ticket()
        ticketing.next_lgkm_ticket()

        # Compute VMEM wait
        vmem_threshold = ticketing.compute_vmem_wait(0)
        assert vmem_threshold == 0

        # LGKM threshold should be unaffected
        assert ticketing._lgkm_last_wait_threshold is None

    def test_lgkm_wait_does_not_affect_vmem(self):
        """Test LGKM wait computation doesn't affect VMEM."""
        ticketing = Ticketing()

        # Issue both types
        ticketing.next_vmem_ticket()
        ticketing.next_lgkm_ticket()

        # Compute LGKM wait
        lgkm_threshold = ticketing.compute_lgkm_wait(0)
        assert lgkm_threshold == 0

        # VMEM threshold should be unaffected
        assert ticketing._vmem_last_wait_threshold is None


class TestEmitterDrivenTicketing:
    """Tests for automatic ticketing via emit_instruction."""

    def test_vmem_instruction_issues_ticket(self):
        """Test VMEM instructions automatically issue tickets."""
        from wave_lang.kernel.wave.asm.asm_emitter import AsmEmitter
        from wave_lang.kernel.wave.asm.instructions import Instruction

        emitter = AsmEmitter(targetid="gfx942", codeobj="COv5")

        # Initially no tickets issued
        assert emitter.ticketing._vmem_last_ticket == -1

        # Emit a VMEM instruction
        class MockBufferLoad(Instruction):
            def __init__(self):
                super().__init__("buffer_load_dwordx4", ["v[0:3]", "v4", "s[0:3]", "0"])

        emitter.emit_instruction(MockBufferLoad())

        # Should have issued one VMEM ticket
        assert emitter.ticketing._vmem_last_ticket == 0
        assert emitter.ticketing._lgkm_last_ticket == -1  # LGKM unchanged

    def test_lgkm_instruction_issues_ticket(self):
        """Test LGKM instructions automatically issue tickets."""
        from wave_lang.kernel.wave.asm.asm_emitter import AsmEmitter
        from wave_lang.kernel.wave.asm.instructions import Instruction

        emitter = AsmEmitter(targetid="gfx942", codeobj="COv5")

        # Initially no tickets issued
        assert emitter.ticketing._lgkm_last_ticket == -1

        # Emit an LGKM instruction (LDS read)
        class MockDSRead(Instruction):
            def __init__(self):
                super().__init__("ds_read_b64", ["v[0:1]", "v2"])

        emitter.emit_instruction(MockDSRead())

        # Should have issued one LGKM ticket
        assert emitter.ticketing._lgkm_last_ticket == 0
        assert emitter.ticketing._vmem_last_ticket == -1  # VMEM unchanged

    def test_salu_instruction_no_ticket(self):
        """Test SALU instructions don't issue tickets."""
        from wave_lang.kernel.wave.asm.asm_emitter import AsmEmitter
        from wave_lang.kernel.wave.asm.instructions import Instruction

        emitter = AsmEmitter(targetid="gfx942", codeobj="COv5")

        # Emit a SALU instruction
        class MockSMov(Instruction):
            def __init__(self):
                super().__init__("s_mov_b32", ["s0", "s1"])

        emitter.emit_instruction(MockSMov())

        # No tickets should be issued
        assert emitter.ticketing._vmem_last_ticket == -1
        assert emitter.ticketing._lgkm_last_ticket == -1

    def test_valu_instruction_no_ticket(self):
        """Test VALU instructions don't issue tickets."""
        from wave_lang.kernel.wave.asm.asm_emitter import AsmEmitter
        from wave_lang.kernel.wave.asm.instructions import Instruction

        emitter = AsmEmitter(targetid="gfx942", codeobj="COv5")

        # Emit a VALU instruction
        class MockVAdd(Instruction):
            def __init__(self):
                super().__init__("v_add_u32", ["v0", "v1", "v2"])

        emitter.emit_instruction(MockVAdd())

        # No tickets should be issued
        assert emitter.ticketing._vmem_last_ticket == -1
        assert emitter.ticketing._lgkm_last_ticket == -1

    def test_mfma_instruction_no_ticket(self):
        """Test MFMA instructions don't issue tickets."""
        from wave_lang.kernel.wave.asm.asm_emitter import AsmEmitter
        from wave_lang.kernel.wave.asm.instructions import Instruction

        emitter = AsmEmitter(targetid="gfx942", codeobj="COv5")

        # Emit an MFMA instruction
        class MockMFMA(Instruction):
            def __init__(self):
                super().__init__(
                    "v_mfma_f32_16x16x16_f16", ["a[0:3]", "v[0:1]", "v[2:3]", "a[0:3]"]
                )

        emitter.emit_instruction(MockMFMA())

        # No tickets should be issued
        assert emitter.ticketing._vmem_last_ticket == -1
        assert emitter.ticketing._lgkm_last_ticket == -1

    def test_mixed_instruction_sequence(self):
        """Test mixed sequence issues tickets correctly."""
        from wave_lang.kernel.wave.asm.asm_emitter import AsmEmitter
        from wave_lang.kernel.wave.asm.instructions import Instruction

        emitter = AsmEmitter(targetid="gfx942", codeobj="COv5")

        # Create mock instructions
        class MockBufferLoad(Instruction):
            def __init__(self):
                super().__init__("buffer_load_dwordx4", ["v[0:3]", "v4", "s[0:3]", "0"])

        class MockDSRead(Instruction):
            def __init__(self):
                super().__init__("ds_read_b64", ["v[0:1]", "v2"])

        class MockVAdd(Instruction):
            def __init__(self):
                super().__init__("v_add_u32", ["v0", "v1", "v2"])

        class MockBufferStore(Instruction):
            def __init__(self):
                super().__init__("buffer_store_dword", ["v0", "v1", "s[0:3]", "0"])

        class MockDSWrite(Instruction):
            def __init__(self):
                super().__init__("ds_write_b32", ["v0", "v1"])

        # Emit sequence: VMEM load, LGKM read, VALU, VMEM store, LGKM write
        emitter.emit_instruction(MockBufferLoad())  # VMEM ticket 0
        assert emitter.ticketing._vmem_last_ticket == 0
        assert emitter.ticketing._lgkm_last_ticket == -1

        emitter.emit_instruction(MockDSRead())  # LGKM ticket 0
        assert emitter.ticketing._vmem_last_ticket == 0
        assert emitter.ticketing._lgkm_last_ticket == 0

        emitter.emit_instruction(MockVAdd())  # No ticket
        assert emitter.ticketing._vmem_last_ticket == 0
        assert emitter.ticketing._lgkm_last_ticket == 0

        emitter.emit_instruction(MockBufferStore())  # VMEM ticket 1
        assert emitter.ticketing._vmem_last_ticket == 1
        assert emitter.ticketing._lgkm_last_ticket == 0

        emitter.emit_instruction(MockDSWrite())  # LGKM ticket 1
        assert emitter.ticketing._vmem_last_ticket == 1
        assert emitter.ticketing._lgkm_last_ticket == 1

    def test_scalar_load_issues_lgkm_ticket(self):
        """Test scalar load (s_load) instructions issue LGKM tickets."""
        from wave_lang.kernel.wave.asm.asm_emitter import AsmEmitter
        from wave_lang.kernel.wave.asm.instructions import Instruction

        emitter = AsmEmitter(targetid="gfx942", codeobj="COv5")

        # Emit a scalar load
        class MockSLoad(Instruction):
            def __init__(self):
                super().__init__("s_load_dwordx2", ["s[0:1]", "s[2:3]", "0x0"])

        emitter.emit_instruction(MockSLoad())

        # Should have issued one LGKM ticket
        assert emitter.ticketing._lgkm_last_ticket == 0
        assert emitter.ticketing._vmem_last_ticket == -1
