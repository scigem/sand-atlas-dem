# Agent instructions — Sand Atlas DEM

Read README.md, IMPLEMENTATION_PLAN.md, docs/STATUS.md and docs/CONTACT_METHOD.md first.
This is a separate repository; do not edit the adjacent atlases as part of solver work.

- Run GPU compilation and execution serially. No concurrent GPU tests or benchmarks.
- Use the explicit Metal backend. Never replace failed GPU work with a CPU simulation.
- Freeze references, workloads and tolerances before executing a study. Preserve failed
  reports and give revised studies new identities; never retune a gate to get a pass.
- Distinguish software execution, independent numerical verification and scientific
  admission. Benchmark throughput does not verify equations or validate sand behaviour.
- Retain source identities, coordinate units, hashes and preprocessing decisions.
  No silent geometry repair, contact truncation or inferred mineral contact properties.
- Keep downloaded geometry and raw publications private unless redistribution is reviewed.
- Local work only. Do not push, deploy or trigger remote CI without explicit instruction.
- Only stage or commit owned files with an explicit request; preserve other agents' work.
