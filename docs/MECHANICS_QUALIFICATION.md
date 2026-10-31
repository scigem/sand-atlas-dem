# Engineering mechanics qualification

Frozen protocols live in `studies/`; executable study code is
`src/sand_atlas_dem/qualification.py`. Reproduce serially with:

```
.venv/bin/sand-dem qualify studies/mechanics-v2.json --output runs/mechanics
.venv/bin/sand-dem qualify studies/mechanics-fast-rotation-v1.json --output runs/rotation
```

Reports retain device identity, protocol hash, Python and shader hashes, workload,
metrics, numerical gates and exceptions. No CPU simulation fallback is used.
The independent FP64 reference is only an oracle for small-case verification.

## Results and scope

`evidence/mechanics-v5/report.json` passes all seven cases:

- Common rigid rotation of a preloaded two-grain contact through 0.8 rad: preserved
  contact keys, approximately 3.1e-5 relative force error and 6.6e-5 history error
  at the finest prescribed-motion increment. This checks discrete frame transport.
- Separation deletes active keys; contact return starts with zero slip.
- Instantaneous nominal pure rolling matches independent nodal FP64 forces,
  torques and slip. Finite-area patches can have creep; zero slip everywhere is not
  imposed as an incorrect point-contact oracle.
- Contact capacity begins at the allocator minimum and grows through explicit
  rollback/retry. The final 124-contact state matches a generous-buffer reference.
- A strongly anisotropic torque-free solid matches a tight DOP853 quaternion ODE
  reference, keeps spatial angular momentum unchanged, and has bounded energy drift.
- Contact-free diagonal affine motion matches exact peculiar velocity evolution;
  integrated cell work agrees with kinetic-energy change within the frozen 0.1% bound.
- Frictionless two-sphere release in a static cell remains within the frozen 2%
  total kinetic-plus-spring-proxy bound for all three timesteps over 20 µs.

The higher-spin `mechanics-fast-rotation-v1` study raises initial angular velocity
by ten to resolve truncation error above the FP32 roundoff floor. Over the same
2.5 µs interval, errors are 4.25e-4, 1.02e-4 and 1.70e-5 rad as the timestep is halved
from 5 ns to 1.25 ns: a 25-fold coarse-to-fine improvement. Its explicit minimum
refinement ratio of three passes. Spatial angular momentum remains unchanged.

The lower-spin and static-cell energy studies show increasing error at the smallest
steps because FP32 accumulation becomes material. A bounded-error pass does not
prove monotone timestep convergence for those cases or exact energy conservation.
The high-spin test supplies a separate resolved integration-convergence case.
Moving contact patches, full assemblies and long-time work balance still require
further studies; these tests do not admit a Hostun material model or reproduce a
published assembly benchmark.

## Retained failures and fixes

`mechanics-v1` failed buffer growth: doubling a requested contacts-per-particle value
could leave the effective allocation stuck at its minimum. Growth now doubles the
actual allocation, and the unchanged gate passes in `mechanics-v2`.
`mechanics-v3` added rolling. Its `passed` NumPy boolean serialized as a string under
the original generic JSON fallback. `mechanics-v4` exposed a separate CLI printing
failure on that scalar type. The serializer now preserves NumPy scalar types and the
study validates/normalizes each gate to a boolean. `mechanics-v5` exits successfully
and contains actual JSON booleans. Earlier reports/logs retain their original bytes.

The software regression suite recorded 22 passed in
`evidence/verification-qualification-v1.log`, including dedicated tests for effective
minimum growth and false boolean serialization. Additional workload construction and
controller-state checks are verified separately when integrated.

## Requalification with compensated FP32 accumulation

The prior reports retain their original identities. After correcting tiny state
increments (see PRECISION.md), `mechanics-compensated-v1` passes all seven cases
against the unchanged mechanics-v2 protocol. Low-spin orientation errors at the
three timesteps are 4.14e-7, 9.61e-8 and 4.61e-8 rad, with constant spatial angular
momentum. Static-cell energy-proxy drift is 7.778e-5, 7.677e-5 and 7.643e-5. The
remaining floor includes geometry/quadrature and the lack of an exact proxy potential.

`mechanics-fast-compensated-v1` repeats the frozen fast-rotation protocol: orientation
errors are 4.286e-4, 1.0504e-4 and 2.6811e-5 rad (15.99-fold coarse/fine improvement).
Spatial angular momentum is unchanged. These results qualify the current solver
within the frozen engineering workloads; full-assembly studies remain separate.


## Extended prepared-assembly study

The frozen `prepared-sphere-assembly-work-v1` protocol and
`examples/assembly_work_study.py` start independent initial-value cases from the
hash-pinned, audited 216-sphere preparation. Three timesteps cover static release,
drag relaxation and cyclic isotropic affine loading. The unchanged 2% engineering
proxy-balance bound is accompanied by an independent sampled stress-power work
audit (0.1% of initial energy), nonnegative dissipation and 1% fine/middle endpoint
stress/energy agreement. Initial contacts, preparation identity and 1% overlap
guard remain required. All nine cases and all three endpoint comparisons pass.
Worst proxy-balance residual is 0.05120%; independent stress-power work discrepancy
is 0.0001164% of initial energy. At least 559 grain pairs remain active, and maximum
overlap is 0.70184%, below the unchanged 1% guard. Fine/middle stress and energy
differences are at most 2.152e-5 and 6.901e-6, respectively, below 1%. The three
independent analytical pressure-volume/time-order checks pass as well. The study
report and prospective source/input freeze are retained in
`evidence/prepared-sphere-assembly-work-v1/`.

History JSONL records are private. `study_time_s` is the physical clock for each
new initial-value case; inherited engine step counters and `elapsed_time` are not
a physical time continuation across changed timesteps. All work/loss counters are
differenced from the restored initial observation. The spring energy is still a
proxy, with no exact potential asserted. Sphere assembly coverage cannot qualify
Hostun quadrature or material behaviour.

The source-supported published assembly candidate and current access/capability
gaps are recorded in [PUBLISHED_ASSEMBLY_REFERENCE.md](PUBLISHED_ASSEMBLY_REFERENCE.md).
