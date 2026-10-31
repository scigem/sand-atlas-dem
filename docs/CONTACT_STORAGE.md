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
