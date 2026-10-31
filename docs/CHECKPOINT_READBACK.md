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
contact capacities, full device rollback copies and checkpoint restore's
capacity-sized host initialization remain separate optimization work. Additional
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
