# Active-contact checkpoint readback

Updated 2026-11-01. This change reduces checkpoint host-transfer memory, not the
solver's fixed-capacity device contact/history storage.

Previously `Engine.state()` read the entire allocated `meta`, `slip` and
`normalArea` buffers and then sliced to the active count. Each returned NumPy view
retained its full underlying allocation. With 50,000 particles and 4,096 contact
slots per particle, those three arrays occupy 9,830,400,000 bytes regardless of
how few contacts are active. A full-buffer readback also created temporary host
copies. This matters independently of the device-memory constraint described in
[HOSTUN_REFERENCE_EXTENSION.md](HOSTUN_REFERENCE_EXTENSION.md).

`Metal.read_prefix` now copies only the requested active scalar prefix to a
bounded staging buffer before host readback. `Engine.state()` requests four words
per active contact for each history array. Empty history produces typed `(0, 4)`
arrays without allocating a staging buffer or submitting work. Invalid counts,
capacity overruns, object dtypes and non-word-aligned requests fail before
allocation. Returned arrays own their host storage independently of the source.

Checkpoint names, dtypes, order and values remain unchanged. No contact key,
friction history, numerical shader, force law or timestep is modified. Device
contact capacities and full device rollback copies remain separate optimization
work. The later prefix-upload change below addresses restore host initialization. Additional
staging memory is proportional to active contacts; no measured peak-memory or
throughput improvement is claimed yet.

CPU transfer tests use synthetic byte buffers that forbid full-capacity host
reads. They verify exact integer/float transfers, independent returned values,
empty prefixes and pre-allocation rejection. Together with prepared-work and
reference-overlap contracts, 24 tests pass (`active-prefix-readback-cpu-v1`).
The complete CPU integration passes 198 tests, skips eight optional cases and
deselects 55 Metal cases (`active-prefix-cpu-integration-v1`); frozen inputs are
unchanged. These tests validate the host transfer contract, not a Metal driver.

The actual-device `test_active_prefix_readback_matches_original_buffer` is queued
with the normal serial Metal suite. It compares exact integer bits, float views,
empty and full prefixes, and verifies the source remains unchanged. Existing
checkpoint/restart tests cover continued simulation after restoration. Execute
these only after the active preparation releases the device. The running
preparation and its future work studies keep their original archived backend and
solver, so they do not silently adopt this readback method.

## Active-contact restore upload

Restore previously created a capacity-sized zero array for each history field,
filled the active prefix and uploaded the whole capacity. `Metal.write_prefix`
now stages and uploads only contiguous active bytes. Empty input allocates and
submits nothing. Capacity overruns, object arrays and non-word-aligned input fail
before device work. Noncontiguous numeric inputs are copied into contiguous
staging without changing their values. Restore also rejects negative contact
counts and mismatched active-history array shapes.

The inactive destination tail is intentionally untouched. This relies on the
existing shader contract: `buildHistoryHash` only inserts IDs below the active
count after clearing the complete destination hash; `resolveContacts` uses that
hash to reach old history and processes only new active IDs. New contacts emit
their metadata and normals before resolution. No inactive tail entry is a valid
lookup target. The frozen shader source is retained in the new evidence archive.

`active-prefix-upload-cpu-integration-v1` passes 204 CPU tests (eight skipped, 57
Metal cases deselected), including exact prefix bytes, unchanged sentinel tails,
noncontiguous arrays, empty inputs and invalid input rejection. The device suite
now additionally checks exact prefix uploads and continuation after restoring
into buffers whose unused metadata is invalid and whose unused float history is
NaN. Those actual-Metal tests remain pending while preparation runs.

Neither transfer optimization is substituted into the active preparation or
its archived-solver work studies. No reduced device-memory or measured application
peak claim is made; these changes remove the full-capacity host initialization
and transfer mechanism in current software.
