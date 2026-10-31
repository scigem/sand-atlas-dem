"""Byte-copy contracts for active-history rollback; no simulated DEM forces."""

import numpy as np
import pytest

from sand_atlas_dem.gpu import Engine


class Buffer:
    def __init__(self, size):
        self.data = np.full(size, 0xA5, np.uint8)
        self.size = size


class Transfer:
    def __init__(self, count):
        self.count = count
        self.allocations = []
        self.copies = []
        self.fail_allocation = False

    def read(self, buffer, dtype):
        return np.array([0, 0, self.count], dtype)

    def buffer(self, *, count):
        if self.fail_allocation:
            raise MemoryError("injected allocation failure")
        self.allocations.append(count * 4)
        return Buffer(count * 4)

    def copy_buffer(self, destination, destination_offset, source, source_offset, size):
        assert size > 0
        assert destination_offset + size <= destination.size
        assert source_offset + size <= source.size
        self.copies.append(size)
        destination.data[destination_offset : destination_offset + size] = source.data[
            source_offset : source_offset + size
        ]


def fixture(count=3):
    engine = Engine.__new__(Engine)
    engine.m = Transfer(count)
    engine.contact_capacity = 20
    engine.buffers = {"control": object()}
    engine.history_backup = {
        name: Buffer(16) for name in ("meta", "slip", "normalArea")
    }
    engine.history_backup["hash"] = Buffer(64)
    committed = {name: Buffer(320) for name in ("meta", "slip", "normalArea")}
    committed["hash"] = Buffer(64)
    for index, buffer in enumerate(committed.values()):
        buffer.data[:] = (np.arange(buffer.size, dtype=np.uint16) + index).astype(
            np.uint8
        )
    return engine, committed


@pytest.mark.parametrize("count", [0, 1, 3, 20])
def test_rollback_restores_exact_active_rows_and_hash_without_unused_tail(count):
    engine, committed = fixture(count)
    originals = {name: b.data.copy() for name, b in committed.items()}
    reserved = engine._reserve_history_backup()
    engine._copy_history_backup(engine.m, committed, reserved)
    # Represents arbitrary overwrites after an encoded chunk, including old tails.
    for buffer in committed.values():
        buffer.data[:] = 0xEE
    engine._copy_history_backup(engine.m, committed, reserved, restore=True)
    for name, buffer in committed.items():
        size = buffer.size if name == "hash" else count * 16
        np.testing.assert_array_equal(buffer.data[:size], originals[name][:size])
        np.testing.assert_array_equal(buffer.data[size:], 0xEE)
    assert engine.m.copies == ([count * 16] * 3 if count else []) + [64] + (
        [count * 16] * 3 if count else []
    ) + [64]


def test_growth_reuses_high_water_buffers_but_copies_only_current_prefix():
    engine, committed = fixture(12)
    engine._reserve_history_backup()
    first = dict(engine.history_backup)
    engine.m.count = 2
    count = engine._reserve_history_backup()
    assert all(engine.history_backup[k] is v for k, v in first.items())
    engine._copy_history_backup(engine.m, committed, count)
    assert engine.m.copies == [32, 32, 32, 64]
    assert engine.m.allocations == [192, 192, 192]


@pytest.mark.parametrize("count", [-1, 21])
def test_invalid_committed_count_rejects_before_allocating(count):
    engine, _ = fixture(count)
    with pytest.raises((ValueError, OverflowError)):
        engine._reserve_history_backup()
    assert engine.m.allocations == []


def test_allocation_failure_preserves_existing_backup_references():
    engine, _ = fixture(20)
    before = dict(engine.history_backup)
    engine.m.fail_allocation = True
    with pytest.raises(MemoryError):
        engine._reserve_history_backup()
    assert engine.history_backup == before


def test_zero_step_does_not_reserve_or_mutate():
    engine = Engine.__new__(Engine)
    engine._reserve_history_backup = lambda: pytest.fail("zero-step reservation")
    engine._advance_once(0)
