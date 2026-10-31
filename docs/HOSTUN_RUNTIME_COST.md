# Refined Hostun preparation cost update

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
