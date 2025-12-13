# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for latency infrastructure: database, provider, and scoreboard.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from wave_lang.kernel.wave.asm.latency_database import (
    LatencyDatabaseManager,
    ArchitectureLatencyDB,
    InstructionTiming,
    MeasurementSource,
)
from wave_lang.kernel.wave.asm.latency_provider import LatencyProvider
from wave_lang.kernel.wave.asm.scoreboard import (
    Scoreboard,
    InstructionCategory,
    PendingInstruction,
)


class TestLatencyDatabase:
    """Tests for LatencyDatabaseManager and ArchitectureLatencyDB."""

    def test_instruction_timing_creation(self):
        """Test creating InstructionTiming instances."""
        timing = InstructionTiming(
            latency_cycles=8.0,
            throughput_ops_per_cycle=0.125,
            measurement_source=MeasurementSource.ISA_MANUAL,
            notes="Test timing",
            stddev_latency=0.5,
        )

        assert timing.latency_cycles == 8.0
        assert timing.throughput_ops_per_cycle == 0.125
        assert timing.measurement_source == MeasurementSource.ISA_MANUAL
        assert timing.notes == "Test timing"
        assert timing.stddev_latency == 0.5

    def test_instruction_timing_serialization(self):
        """Test InstructionTiming to_dict and from_dict."""
        timing = InstructionTiming(
            latency_cycles=4.0,
            throughput_ops_per_cycle=1.0,
            measurement_source=MeasurementSource.LLVM_CODEGEN,
            notes="LLVM pattern",
            stddev_latency=0.0,
        )

        # Serialize
        d = timing.to_dict()
        assert d["latency_cycles"] == 4.0
        assert d["measurement_source"] == "llvm_codegen"

        # Deserialize
        timing2 = InstructionTiming.from_dict(d)
        assert timing2.latency_cycles == timing.latency_cycles
        assert timing2.measurement_source == timing.measurement_source

    def test_architecture_db_creation(self):
        """Test creating ArchitectureLatencyDB."""
        db = ArchitectureLatencyDB(
            architecture="gfx942",
            version="1.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            instructions={},
            instruction_classes={},
        )

        assert db.architecture == "gfx942"
        assert db.version == "1.0"
        assert len(db.instructions) == 0
        assert len(db.instruction_classes) == 0

    def test_architecture_db_direct_assignment(self):
        """Test directly assigning instructions to database."""
        db = ArchitectureLatencyDB(
            architecture="gfx942",
            version="1.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            instructions={},
            instruction_classes={},
        )

        # Direct dict assignment
        db.instructions["v_add_u32"] = InstructionTiming(
            latency_cycles=4.0,
            throughput_ops_per_cycle=0.25,
            measurement_source=MeasurementSource.ISA_MANUAL,
            notes="Vector add",
            stddev_latency=0.0,
        )

        assert "v_add_u32" in db.instructions
        assert db.instructions["v_add_u32"].latency_cycles == 4.0

    def test_architecture_db_get_timing(self):
        """Test querying instruction timing."""
        db = ArchitectureLatencyDB(
            architecture="gfx942",
            version="1.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            instructions={
                "buffer_load_dwordx4": InstructionTiming(
                    latency_cycles=80.0,
                    throughput_ops_per_cycle=0.25,
                    measurement_source=MeasurementSource.ESTIMATED,
                    notes="",
                    stddev_latency=0.0,
                )
            },
            instruction_classes={},
        )

        # Direct lookup
        timing = db.get_timing("buffer_load_dwordx4")
        assert timing is not None
        assert timing.latency_cycles == 80.0

        # Not found returns None
        timing = db.get_timing("unknown_instruction")
        assert timing is None

    def test_architecture_db_save_load(self):
        """Test saving and loading database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create database
            db = ArchitectureLatencyDB(
                architecture="gfx942",
                version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                instructions={
                    "v_add_u32": InstructionTiming(
                        latency_cycles=4.0,
                        throughput_ops_per_cycle=0.25,
                        measurement_source=MeasurementSource.ISA_MANUAL,
                        notes="Test",
                        stddev_latency=0.0,
                    )
                },
                instruction_classes={},
            )

            # Save
            db_path = Path(tmpdir) / "gfx942.json"
            db.save(str(db_path))
            assert db_path.exists()

            # Load
            loaded_db = ArchitectureLatencyDB.load(str(db_path))
            assert loaded_db.architecture == "gfx942"
            assert "v_add_u32" in loaded_db.instructions
            assert loaded_db.instructions["v_add_u32"].latency_cycles == 4.0

    def test_database_manager_load_save(self):
        """Test LatencyDatabaseManager operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LatencyDatabaseManager(db_root=tmpdir)

            # Create and save
            db = ArchitectureLatencyDB(
                architecture="gfx942",
                version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                instructions={
                    "v_add_u32": InstructionTiming(
                        latency_cycles=4.0,
                        throughput_ops_per_cycle=0.25,
                        measurement_source=MeasurementSource.ISA_MANUAL,
                        notes="",
                        stddev_latency=0.0,
                    )
                },
                instruction_classes={},
            )
            manager.save_architecture("gfx942", db)

            # Load
            loaded_db = manager.load_architecture("gfx942")
            assert loaded_db.architecture == "gfx942"
            assert "v_add_u32" in loaded_db.instructions


class TestLatencyProvider:
    """Tests for LatencyProvider."""

    @pytest.fixture
    def test_db_path(self):
        """Create a test database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ArchitectureLatencyDB(
                architecture="gfx942",
                version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                instructions={
                    "v_add_u32": InstructionTiming(
                        latency_cycles=4.0,
                        throughput_ops_per_cycle=0.25,
                        measurement_source=MeasurementSource.ISA_MANUAL,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "buffer_load_dwordx4": InstructionTiming(
                        latency_cycles=80.0,
                        throughput_ops_per_cycle=0.25,
                        measurement_source=MeasurementSource.ESTIMATED,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "buffer_load_dwordx2": InstructionTiming(
                        latency_cycles=80.0,
                        throughput_ops_per_cycle=0.25,
                        measurement_source=MeasurementSource.ESTIMATED,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "v_mfma_f32_16x16x16_f16": InstructionTiming(
                        latency_cycles=8.0,
                        throughput_ops_per_cycle=0.125,
                        measurement_source=MeasurementSource.ISA_MANUAL,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "mfma_to_agpr_read": InstructionTiming(
                        latency_cycles=6.0,
                        throughput_ops_per_cycle=1.0,
                        measurement_source=MeasurementSource.LLVM_CODEGEN,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "s_mov_b32": InstructionTiming(
                        latency_cycles=1.0,
                        throughput_ops_per_cycle=1.0,
                        measurement_source=MeasurementSource.LLVM_CODEGEN,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "ds_read_b64": InstructionTiming(
                        latency_cycles=20.0,
                        throughput_ops_per_cycle=0.5,
                        measurement_source=MeasurementSource.ESTIMATED,
                        notes="",
                        stddev_latency=0.0,
                    ),
                },
                instruction_classes={},
            )

            db_path = Path(tmpdir) / "gfx942.json"
            db.save(str(db_path))

            yield tmpdir

    def test_provider_initialization(self, test_db_path):
        """Test LatencyProvider initialization."""
        provider = LatencyProvider(arch="gfx942")
        # Provider uses default path, so create one there too
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ArchitectureLatencyDB(
                architecture="gfx942",
                version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                instructions={},
                instruction_classes={},
            )
            db_path = Path(tmpdir) / "gfx942.json"
            db.save(str(db_path))

            manager = LatencyDatabaseManager(db_root=tmpdir)
            loaded = manager.load_architecture("gfx942")
            assert loaded.architecture == "gfx942"

    def test_get_latency(self, test_db_path):
        """Test querying instruction latency."""
        # Need to create provider with test db path
        # Since LatencyProvider uses hardcoded path, we test the database manager instead
        manager = LatencyDatabaseManager(db_root=test_db_path)
        db = manager.load_architecture("gfx942")

        # Known instruction
        timing = db.get_timing("v_add_u32")
        assert timing is not None
        assert timing.latency_cycles == 4.0

        # MFMA instruction
        timing = db.get_timing("v_mfma_f32_16x16x16_f16")
        assert timing is not None
        assert timing.latency_cycles == 8.0

        # Hazard latency
        timing = db.get_timing("mfma_to_agpr_read")
        assert timing is not None
        assert timing.latency_cycles == 6.0

        # Unknown instruction returns None
        timing = db.get_timing("unknown_instruction")
        assert timing is None

    def test_get_throughput(self, test_db_path):
        """Test querying instruction throughput."""
        manager = LatencyDatabaseManager(db_root=test_db_path)
        db = manager.load_architecture("gfx942")

        timing = db.get_timing("v_add_u32")
        assert timing is not None
        assert timing.throughput_ops_per_cycle == 0.25

        timing = db.get_timing("v_mfma_f32_16x16x16_f16")
        assert timing is not None
        assert timing.throughput_ops_per_cycle == 0.125

        # Unknown instruction returns None
        timing = db.get_timing("unknown_instruction")
        assert timing is None


class TestScoreboard:
    """Tests for Scoreboard-based hazard detection."""

    @pytest.fixture
    def test_provider(self):
        """Create a test LatencyProvider with known latencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ArchitectureLatencyDB(
                architecture="gfx942",
                version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                instructions={
                    "buffer_load_dwordx4": InstructionTiming(
                        latency_cycles=80.0,
                        throughput_ops_per_cycle=0.25,
                        measurement_source=MeasurementSource.ESTIMATED,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "v_add_u32": InstructionTiming(
                        latency_cycles=4.0,
                        throughput_ops_per_cycle=0.25,
                        measurement_source=MeasurementSource.ISA_MANUAL,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "v_mfma_f32_16x16x16_f16": InstructionTiming(
                        latency_cycles=8.0,
                        throughput_ops_per_cycle=0.125,
                        measurement_source=MeasurementSource.ISA_MANUAL,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "ds_read_b64": InstructionTiming(
                        latency_cycles=20.0,
                        throughput_ops_per_cycle=0.5,
                        measurement_source=MeasurementSource.ESTIMATED,
                        notes="",
                        stddev_latency=0.0,
                    ),
                    "s_mov_b32": InstructionTiming(
                        latency_cycles=1.0,
                        throughput_ops_per_cycle=1.0,
                        measurement_source=MeasurementSource.LLVM_CODEGEN,
                        notes="",
                        stddev_latency=0.0,
                    ),
                },
                instruction_classes={},
            )

            db_path = Path(tmpdir) / "gfx942.json"
            db.save(str(db_path))

            # Create provider - note it uses hardcoded path, so we work around it
            manager = LatencyDatabaseManager(db_root=tmpdir)
            provider = LatencyProvider.__new__(LatencyProvider)
            provider.arch = "gfx942"
            provider.db_manager = manager
            provider.db = manager.load_architecture("gfx942")
            yield provider

    def test_scoreboard_initialization(self, test_provider):
        """Test Scoreboard initialization."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        assert scoreboard.current_cycle == 0
        assert len(scoreboard.pending[InstructionCategory.VMEM]) == 0

    def test_instruction_categorization(self, test_provider):
        """Test instruction category inference."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        assert (
            scoreboard._categorize_instruction("buffer_load_dwordx4")
            == InstructionCategory.VMEM
        )
        assert (
            scoreboard._categorize_instruction("buffer_store_dword")
            == InstructionCategory.VMEM
        )
        assert (
            scoreboard._categorize_instruction("ds_read_b64")
            == InstructionCategory.LGKM
        )
        assert (
            scoreboard._categorize_instruction("s_load_dwordx2")
            == InstructionCategory.LGKM
        )
        assert (
            scoreboard._categorize_instruction("v_mfma_f32_16x16x16_f16")
            == InstructionCategory.MFMA
        )
        assert (
            scoreboard._categorize_instruction("v_add_u32") == InstructionCategory.VALU
        )
        assert (
            scoreboard._categorize_instruction("s_mov_b32") == InstructionCategory.SALU
        )

    def test_issue_instruction(self, test_provider):
        """Test issuing instructions to scoreboard."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        pending = scoreboard.issue(
            "buffer_load_dwordx4", writes={"v0", "v1", "v2", "v3"}, ticket=0
        )

        assert pending.instruction == "buffer_load_dwordx4"
        assert pending.category == InstructionCategory.VMEM
        assert pending.latency == 80.0
        assert pending.writes == {"v0", "v1", "v2", "v3"}
        assert pending.ticket == 0

        # Check it's in pending list
        assert len(scoreboard.pending[InstructionCategory.VMEM]) == 1

    def test_advance_cycles(self, test_provider):
        """Test advancing scoreboard and retiring instructions."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        # Issue a short-latency instruction
        scoreboard.issue("v_add_u32", writes={"v0"})
        assert len(scoreboard.pending[InstructionCategory.VALU]) == 1
        assert scoreboard.current_cycle == 0

        # Advance past latency (4 cycles)
        scoreboard.advance(5)
        assert scoreboard.current_cycle == 5

        # Should be retired
        assert len(scoreboard.pending[InstructionCategory.VALU]) == 0

    def test_raw_hazard_detection(self, test_provider):
        """Test Read-After-Write hazard detection."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        # Issue load that writes to v0-v3
        scoreboard.issue(
            "buffer_load_dwordx4", writes={"v0", "v1", "v2", "v3"}, ticket=0
        )

        # Check hazard for instruction reading v0
        hazard = scoreboard.check_hazard(reads={"v0"})
        assert hazard is not None
        hazard_type, cycles_needed = hazard
        assert hazard_type == "RAW"
        assert cycles_needed == 80  # Full load latency

        # Advance halfway
        scoreboard.advance(40)
        hazard = scoreboard.check_hazard(reads={"v0"})
        assert hazard is not None
        hazard_type, cycles_needed = hazard
        assert cycles_needed == 40  # Remaining latency

        # Advance past latency
        scoreboard.advance(50)
        hazard = scoreboard.check_hazard(reads={"v0"})
        assert hazard is None  # No hazard, instruction complete

    def test_waw_hazard_detection(self, test_provider):
        """Test Write-After-Write hazard detection."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        # Issue instruction writing to v0
        scoreboard.issue("v_add_u32", writes={"v0"})

        # Check hazard for another instruction writing v0
        hazard = scoreboard.check_hazard(writes={"v0"})
        assert hazard is not None
        hazard_type, cycles_needed = hazard
        assert hazard_type == "WAW"
        assert cycles_needed == 4

    def test_no_hazard(self, test_provider):
        """Test case with no hazards."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        # Issue instruction writing v0
        scoreboard.issue("v_add_u32", writes={"v0"})

        # Check for instruction using different registers
        hazard = scoreboard.check_hazard(reads={"v1"}, writes={"v2"})
        assert hazard is None

    def test_vmem_wait_count(self, test_provider):
        """Test VMEM ticket-based wait count calculation."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        # Issue multiple loads with tickets
        scoreboard.issue("buffer_load_dwordx4", ticket=0)
        scoreboard.issue("buffer_load_dwordx4", ticket=1)
        scoreboard.issue("buffer_load_dwordx4", ticket=2)

        # Wait for ticket 0: allows tickets 1 and 2 to be in-flight
        vmcnt = scoreboard.get_vmem_wait_count(0)
        assert vmcnt == 2

        # Wait for ticket 1: allows ticket 2 to be in-flight
        vmcnt = scoreboard.get_vmem_wait_count(1)
        assert vmcnt == 1

        # Wait for ticket 2: all must complete
        vmcnt = scoreboard.get_vmem_wait_count(2)
        assert vmcnt == 0

    def test_lgkm_wait_count(self, test_provider):
        """Test LGKM ticket-based wait count calculation."""
        scoreboard = Scoreboard(latency_provider=test_provider)

        # Issue multiple LDS reads with tickets
        scoreboard.issue("ds_read_b64", ticket=0)
        scoreboard.issue("ds_read_b64", ticket=1)

        # Wait for ticket 0
        lgkmcnt = scoreboard.get_lgkm_wait_count(0)
        assert lgkmcnt == 1  # Ticket 1 can be in-flight

        # Wait for ticket 1
        lgkmcnt = scoreboard.get_lgkm_wait_count(1)
        assert lgkmcnt == 0  # All must complete
