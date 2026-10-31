# Contact storage and the 50k memory requirement

The source-frozen `evidence/contact-storage-ledger-v1` executes only constructor
allocation declarations using symbolic byte-count buffers. It creates neither a
Metal device nor capacity-sized NumPy arrays. Declaration totals agree exactly
with an independent algebraic count at 424, 10k and 50k particles. The unchanged
capacity is 4,096 surface contacts per particle; no dense-capacity adequacy is
inferred from this planning study.

Let C be contact capacity and H the power-of-two hash-table size. Two history
sides store three 16-byte rows per contact each; rollback retains another copy.
The three hash tables use four bytes per slot. Point/depth, force and normal-loss
scratch use 36 bytes per contact; directed adjacency adds eight. Their total is
therefore **188 C + 12 H bytes**, excluding all other solver allocations.

| Contact storage component | 10k particles, GiB | 50k particles, GiB |
| --- | ---: | ---: |
| Two history state sides | 3.662 | 18.311 |
| History state rollback | 1.831 | 9.155 |
| Three hash tables | 1.500 | 6.000 |
| Point/force/loss scratch | 1.373 | 6.866 |
| Directed adjacency | 0.305 | 1.526 |
| Total contact storage | **8.672** | **41.858** |

At 424 particles this same subtotal is 0.351 GiB. These are logical requested
buffer sizes, not measured peak resident memory. Geometry, particle/pair/bin
state, reductions/scans, host arrays, compiler and driver costs remain additional.
The broader conservative constructor estimate remains a separate quantity;
this subtotal does not replace or retroactively revise its retained reports.

The 50k configuration exceeds this Mac's 36 GiB from contact storage alone.
Geometry-only changes cannot resolve that case. Host checkpoint prefix transfers
already reduce host staging but leave this device storage unchanged.

The next implementation should address active-contact storage and rollback
allocation while preserving exact keys, history transport, overflow rejection,
rollback and restart. Prospective tests must cover alternating history sides,
empty and growing active sets, deliberately poisoned unused tails, rejection after
multiple encoded steps, and exact resumed friction/work histories. Capacity
reduction without evidence is not an optimization. Shared hash/scratch layouts
would also need an explicit dispatch-lifetime audit before aliasing allocations.
Any implemented change requires serial actual-Metal verification and new measured
peak accounting. Neither such optimization nor dense 50k fit is claimed here.

## Active-prefix rollback implementation

The subsequent implementation reserves the three history rollback buffers from
the committed active count rather than allocating full contact capacity at
construction. Each begins at one row, grows to a new observed high-water count,
and is reused for smaller active sets. Both backup and restore copy only the
current committed prefix. The complete hash backup remains unchanged, so copied
hash entries retain their exact original indices and bytes. Unused history tails
are unreachable after the original control count/hash is restored.

Reservation happens before cell or wall mutation. Replacement references are
installed together after successful allocation, and invalid counts reject before
allocation. Contact capacity, overflow handling, shaders and contact equations are
unchanged. The conservative preflight estimate still reserves the full possible
history backup; it is not lowered to predict an unobserved active count.

Potential logical savings are 48 times the difference between capacity and the
three-buffer high-water count, with a one-row minimum. Actual dense savings are
unknown, and new allocations can temporarily coexist with previous buffers during
growth. The 41.858 GiB ledger above retains its original frozen eager-allocation
identity and remains the full-capacity upper configuration for these components.

Nine independent CPU byte-copy contracts cover empty/full prefixes, arbitrary
overwrites, untouched unused tails, shrinking active sets, high-water reuse,
allocation failure and invalid counts. Full CPU regression passes 221 tests,
with eight skipped and 59 Metal cases deselected (`active-history-backup-cpu-v1`).
Two new actual-Metal cases cover both history-side parities and an injected
rejection after three encoded steps, checking complete exposed state and exact
hash restoration. They remain queued behind the ongoing immutable preparation.
No actual-device verification, measured saving or dense memory fit is claimed.

The device checks are frozen separately under
`evidence/active-history-backup-metal-v1/source_snapshot`, including the complete
package and selected test file. Eight cases collect successfully without a GPU.
The standard-library runner verifies every frozen source hash before solver
import and rejects execution while the existing preparation manifest is
nonterminal. That rejection was checked and retained. It never overwrites an
earlier execution directory. After verifying that the GPU is free, run serially:

```sh
.venv/bin/python evidence/active-history-backup-metal-v1/run.py --execute
```

The runner's dependency check does not detect unrelated GPU processes; the
operator must still enforce serial execution. Without `--execute` it checks hashes
only. Collection and dependency checks do not establish device-test success.
