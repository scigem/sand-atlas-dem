# Bounded full-inventory Hostun preparation

This is the engineering preparation preceding mesh-assembly work/timestep studies.
It is not the 10,000-grain material pilot or a calibrated constitutive dataset.
The accepted subset has 53 original grains; 424 particles repeat each grain eight
times under seeded balanced sampling. No invalid grain is repaired or reintroduced.

The frozen experiment and acceptance policy are
`studies/hostun-balanced424-preparation-v3.json` and
`studies/hostun-balanced424-preparation-policy-v3.json`. Launch with:

```sh
PYTHONPATH=src python examples/qualified_preparation.py \
  studies/hostun-balanced424-preparation-policy-v3.json \
  --output evidence/hostun-balanced424-preparation-v3
```

This requires complete accepted `hostun-library-gpu-quadrature-adaptive-v3`
qualification, all 106 original reference poses and 212 GPU cases, identical
candidate bytes and matching solver sources. It archives the policy, qualification
reports and source identities before launching the immutable controller. The
actual failed static study is rejected before output creation in
`qualified-preparation-preflight-v1`. Do not bypass this qualification dependency.

Physical values are explicitly declared examples: density 2650 kg/m³, normal and
tangential traction stiffnesses 1e12 and 5e11 N/m³, friction 0.5, pressure 10 kPa,
zero contact viscosity, and 2 ns timestep. None is inferred from the sand name.
Initial solid fraction is 0.01. The initial rate limit is 1000/s, latched down to
50/s after contact, with gain 1000, filter time 10 microseconds and drag 1e5/s.
The computational memory budget is 12 GiB; detected advancement capacity overflow
can restore/grow/retry within that budget, without contact truncation.

There are at most 16 million steps (32 ms physical). Controller observations are
125 steps apart (0.25 microseconds). Acceptance requires 100 consecutive samples
(25 microseconds) with normal stresses within 1%, kinetic/spring-proxy ratio at
most 1e-5 and force imbalance at most 1e-3. The full-history overlap limit remains
1%. The independently frozen audit additionally requires final nominal solid
fraction at least 0.45 and at least 424 active grain pairs throughout the accepted
window. Budget exhaustion is failure, without automatic continuation.

`hostun-balanced424-intake-v3` independently checks the initial periodic bounding
sphere gaps (minimum 0.299662 mm), exact balanced inventory and conservative parent
timestep bound (4.9154 ns). The periodic image ceiling permits nominal solid
fraction 0.65; this is a computational bound, not a prediction of packing density.
The actual adaptive-asset device preflight and dynamic preparation remain required.

Output `study/` contains the frozen controller, private shape asset, chunk recovery
state and final prepared checkpoint or retained failure. The wrapper then runs
`audit_preparation.py` on terminal success; `acceptance-audit.json` binds the exact
checkpoint and shape hashes. The audit checks ordered complete observations,
all-history overlaps, stress/energy/imbalance, source stability and independent
rigid-volume void ratio. Checkpoint formats /3 and /5 reconstruct immutable shape
IDs/scales from the archived initialization function on resume; the auditor checks
that same CPU contract, rather than claiming nonexistent stored ID fields.

The auditor passes on the unchanged prior 216-sphere checkpoint in
`preparation-auditor-controller-v4-v3`. The original aggregate-only audit and the
failed direct-ID-field attempt retain distinct identities. This verifies the
software audit path; it does not qualify the Hostun specimen.

After accepted preparation, freeze mesh-assembly static, drag and affine work
studies, timestep refinement and loading-rate studies using the exact checkpoint.
A prepared packing alone does not qualify dynamic/frictional accuracy or a
constitutive response. The larger material pilot, dense performance study and
independent published/experimental comparison remain separate plan gates.
