# Refined Hostun preparation cost update

The subsequent completed 750k checkpoint retains the preceding trajectory
exactly. Its latest 250k-step interval takes 3210.239 s: mean 12.841 ms/step,
median 12.855 and p95 13.006. Nominal solid fraction is still only 0.066485,
with two active grain pairs. This newer observation does not turn the original
conditional runtime scenarios below into an ETA or a dense benchmark. The exact
checkpoint, source/shape hashes and independent controller/geometric replay are
retained under `hostun-balanced424-750k-observation-v1`. Preparation continues
unchanged; no gate is relaxed and no queued GPU work is started concurrently.

The original short sphere timings do not describe the later 53.6-million-node,
all-53-shape exact-mesh Hostun preparation. The independently reproducible
`hostun-preparation-cost-update-v1` uses the completed 500,000-step checkpoint,
its full diagnostic history and the unchanged preparation policy. It adds no
acceptance threshold and does not alter the live run.

At that checkpoint, nominal solid fraction is 0.0616808. The latched controller
caps compression at 50/s along each axis. For fixed rigid total grain volume,
`d(log(phi))/dt <= 3 * 50/s` in the continuous model. Therefore the additional
physical time to the mandatory nominal density floor 0.45 is at least
`log(0.45 / 0.0616808) / 150 = 0.0132485 s`, ignoring arithmetic roundoff. At the
declared 2 ns timestep this is approximately 6.624 million additional steps.
The first diagnostic that could reach that density is around step 7.124 million;
this is not a prediction that it will reach it then.

The remaining frozen budget is 15.5 million steps (31 ms physical). No automatic
extension or rate increase is introduced. The density floor is only one
requirement; pressure, contact inventory and 100 consecutive equilibrium samples
remain necessary and are not established by the bound.

The completed post-latch interval covers 207,125 steps. Its observed mean cost is
11.720 ms/step, median 12.060 ms/step and p95 12.548 ms/step, including the recorded
advance/diagnostic intervals. Holding those costs fixed for the minimum density
steps gives respectively 21.56, 22.19 and 23.09 additional wall hours. These
conditional scenarios are neither wall-time bounds nor an ETA. The observations
are dilute first-contact coverage, not a sustained dense benchmark. More dense
contacts, changing allocations, thermals, compilation and interruptions can
change future cost materially.

This update supersedes using the small sphere timings as a cost estimate for
this refined mesh study. It does not supersede those sphere measurements or
claim a dense 10k/50k cost. The next performance action is an actual-device
subsystem profile when the current serial run releases the device, followed by
source-frozen optimization and numerical requalification. The actual queued
Metal checkpoint-transfer tests, revised quadrature candidate and dependent
mesh-work study remain separately required. The 50k device-memory constraint
is also unresolved; see [HOSTUN_REFERENCE_EXTENSION.md](HOSTUN_REFERENCE_EXTENSION.md).

## Queued original-solver checkpoint profile

`hostun-first-contact-profile-v1` is now resolved without GPU execution. Its
private checkpoint copy matches the immutable 500k parent exactly. All 32
original preparation source/metadata files are verified and copied; the separately
frozen profiling driver is added alongside them. The current checkout's modified
backend and solver are not substituted. Policy, resolved protocol and every
archived source have receipt hashes.

Three repetitions restore the same state. Each performs 16 ordinary warmup steps
and 128 timestamped steps at 2 ns. The prescribed cell rate is -50/s along each
axis, with the declared global drag 100,000/s. The archived `profile_step` calls
`bindings()` with default zero drag; the driver explicitly supplies the declared
drag through an instance binding wrapper, retaining its original shader/dispatch
sequence. Normal traction viscosity and walls are unsupported by this narrow
profile and are rejected before freezing. This is a prescribed short loading
segment, not continuation of the pressure-filter controller.

The driver records per-step wall time, available GPU timestamps for drift,
neighbours, contact geometry, friction/history, force reduction and final kick,
plus actual buffer/driver memory accounting and the Metal doctor identity. A
missing timestamp result is retained explicitly; wall timings do not become
inferred GPU timings. Single-step timestamp synchronization differs from the
production batched advancement. No sustained, thermal, dense-specimen, material
or preparation acceptance is claimed by this short profile.

Eight CPU resolver checks pass, including exact archive/checkpoint freezing,
changed sources/assets, unsupported rates/budgets and path escapes. A changed
resolved protocol fails before solver import. Full CPU integration passes 212
tests (eight skipped, 57 Metal cases deselected), with frozen sources unchanged.
Actual device profiling remains queued until the continuing preparation releases
the GPU. To execute the already frozen workload then, from the DEM worktree:

```sh
PYTHONPATH=evidence/hostun-first-contact-profile-v1/source_snapshot \
  /Users/bmar5496/code/sand-atlas-dem/.venv/bin/python \
  evidence/hostun-first-contact-profile-v1/source_snapshot/profile_checkpoint.py \
  evidence/hostun-first-contact-profile-v1/protocol.json --execute-frozen \
  --output evidence/hostun-first-contact-profile-v1/profile
```

Do not run that command concurrently with the existing GPU preparation. The
original completed checkpoint and its referenced hash-checked shape asset must
remain available. The output uses a fresh directory and preserves failed or
partial trials. It never alters the active preparation, its checkpoint or gates.
