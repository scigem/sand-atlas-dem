"""Host readback contract; emulated buffers never initialize a GPU device."""

from types import SimpleNamespace

import numpy as np
import pytest

from sand_atlas_dem.backend import Metal


class BytesBuffer:
    def __init__(self, data, *, forbid_read=False):
        self.data = np.array(data, dtype=np.uint8, copy=True)
        self.size = self.data.nbytes
        self.forbid_read = forbid_read

    def to_numpy(self):
        assert not self.forbid_read, "Full-capacity readback is forbidden"
        return self.data


class HostTransfer:
    read = Metal.read
    read_prefix = Metal.read_prefix

    def __init__(self):
        self.allocations = []
        self.copies = []
        self.waits = 0
        self.device = SimpleNamespace(
            create_command_encoder=lambda: self, wait=self.wait
        )

    def buffer(self, *, count):
        self.allocations.append(count * 4)
        return BytesBuffer(np.zeros(count * 4, dtype=np.uint8))

    def copy_buffer(self, dst, dst_offset, src, src_offset, size):
        self.copies.append(size)
        dst.data[dst_offset : dst_offset + size] = src.data[
            src_offset : src_offset + size
        ]

    def submit(self, encoder):
        assert encoder is self

    def wait(self):
        self.waits += 1


@pytest.mark.parametrize("dtype", [np.float32, np.uint32])
def test_prefix_is_exact_independent_and_reads_only_active_bytes(dtype):
    data = np.arange(1024, dtype=dtype)
    source = BytesBuffer(data.view(np.uint8), forbid_read=True)
    transfer = HostTransfer()
    result = transfer.read_prefix(source, 12, dtype, shape=(3, 4))
    np.testing.assert_array_equal(result, data[:12].reshape(3, 4))
    assert result.dtype == dtype
    assert result.flags.owndata or result.base.flags.owndata
    assert transfer.allocations == [48]
    assert transfer.copies == [48]
    assert transfer.waits == 1
    source.data[:] = 0
    np.testing.assert_array_equal(result, data[:12].reshape(3, 4))


def test_empty_prefix_allocates_and_transfers_nothing():
    transfer = HostTransfer()
    result = transfer.read_prefix(
        BytesBuffer(np.zeros(16), forbid_read=True), 0, np.uint32, shape=(0, 4)
    )
    assert result.shape == (0, 4)
    assert result.dtype == np.uint32
    assert not transfer.allocations and not transfer.copies and not transfer.waits


@pytest.mark.parametrize(
    "count,dtype,error",
    [
        (-1, np.float32, "nonnegative"),
        (True, np.float32, "nonnegative"),
        (1.0, np.float32, "nonnegative"),
        (5, np.float32, "capacity"),
        (1, np.uint8, "aligned"),
        (1, object, "fixed-size"),
    ],
)
def test_invalid_prefix_is_rejected_before_allocation(count, dtype, error):
    transfer = HostTransfer()
    with pytest.raises(ValueError, match=error):
        transfer.read_prefix(BytesBuffer(np.zeros(16)), count, dtype)
    assert not transfer.allocations and not transfer.copies
