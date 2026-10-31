"""Fixed-workload benchmarks; timings never stand in for numerical verification."""

import statistics
import time
from pathlib import Path

import numpy as np

from .geometry import ShapeLibrary, sphere
from .gpu import Engine
from .spec import Experiment
from .util import sha256, write_json


def controlled_pairs(library, particles, *, depth_fraction=0.0025, seed=123):
    """Construct isolated contacting mesh pairs; geometry preprocessing, not dynamics.

    Each grain uses its original mesh-derived field and surface nodes. Identical
    shapes within each pair allow controlled penetration without silently creating
    deep overlaps in an irregular assembly. All pairs are randomly oriented.
    """
    from scipy.spatial.transform import Rotation

    if particles % 2 or particles < 2:
        raise ValueError("Controlled-pair workload needs an even particle count")
    separations = []
    for shape in library.shapes:
        diameter = (6 * shape.volume / np.pi) ** (1 / 3)
        target = depth_fraction * diameter

        def depth(distance):
            branch = np.array([distance, 0, 0])
            return max(
                float(-shape.query(shape.samples - branch)[0].min()),
                float(-shape.query(shape.samples + branch)[0].min()),
                0,
            )

        upper = 2.1 * shape.radius
        previous = upper
        for distance in np.linspace(upper, 0, 257)[1:]:
            if depth(distance) >= target:
                lower = distance
                upper = previous
                break
            previous = distance
        else:
            raise ValueError(f"No controlled contact bracket for {shape.id}")
        for _ in range(32):
            middle = (lower + upper) / 2
            if depth(middle) >= target:
                lower = middle
            else:
                upper = middle
        separations.append((lower + upper) / 2)
    count = particles // 2
    ids = np.repeat(np.arange(count, dtype=np.uint32) % len(library.shapes), 2)
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    matrices = Rotation.from_quat(q).as_matrix()
    radius = max(s.radius for s in library.shapes)
    side = int(np.ceil(count ** (1 / 3)))
    spacing = 6 * radius
    indices = np.stack(
        np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), -1
    ).reshape(-1, 3)[:count]
    centers = (indices + 0.5) * spacing
    offsets = matrices[:, :, 0] * np.array(separations)[ids[::2], None] / 2
    positions = np.empty((particles, 3))
    positions[::2] = centers - offsets
    positions[1::2] = centers + offsets
    return (
        positions,
        np.full(3, side * spacing),
        ids,
        np.repeat(q, 2, axis=0).astype(np.float32),
        separations,
    )


def memory_report(engine):
    """Account for all solver-owned buffers and expose driver/host limits."""
    import resource
    import sys

    buffers = {}

    def visit(value):
        if isinstance(value, engine.m.spy.Buffer):
            buffers[id(value)] = value
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for name in (
        "buffers",
        "backups",
        "history_buffers",
        "history_backup",
        "diag_buffers",
        "fabric_buffers",
    ):
        visit(getattr(engine, name))

    def scan(value):
        visit(value.totals)
        visit(value.offsets)
        if value.child:
            scan(value.child)

    scan(engine.scan_pairs)
    scan(engine.scan_contacts)
    report = dict(
        solver_buffer_count=len(buffers),
        logical_solver_buffer_bytes=sum(b.size for b in buffers.values()),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        excludes=[
            "Python/Slang compiler object overhead from logical buffer count",
            "Other processes and system allocations; unified-memory categories can overlap",
        ],
    )
    try:
        report["reported_device_resource_bytes"] = sum(
            b.memory_usage.device for b in buffers.values()
        )
        report["reported_host_resource_bytes"] = sum(
            b.memory_usage.host for b in buffers.values()
        )
    except RuntimeError as exc:
        report["resource_usage_unavailable"] = str(exc)
    try:
        report["internal_heaps"] = [
            dict(
                label=h.label,
                total_allocated=h.total_allocated,
                total_mem_usage=h.total_mem_usage,
                num_allocations=h.num_allocations,
                num_pages=h.num_pages,
            )
            for h in engine.m.device.report_heaps()
        ]
    except RuntimeError as exc:
        report["internal_heaps_unavailable"] = str(exc)
    return report


def benchmark(
    output,
    *,
    particles=1000,
    resolution=32,
    samples=512,
    steps=10,
    repeats=3,
    library_path=None,
    metal=None,
    packing="simple-cubic",
    contact_geometry="sdf",
    timestep=1e-8,
    minimum_seconds=0,
    pairs_per_particle=None,
    contacts_per_particle=None,
):
    if minimum_seconds < 0 or not np.isfinite(minimum_seconds):
        raise ValueError("Finite nonnegative benchmark duration required")
    if packing not in {"simple-cubic", "controlled-pairs"}:
        raise ValueError("Unknown benchmark packing")
    if min(particles, steps, repeats) < 1:
        raise ValueError("Positive particles, steps and repeats required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    library = (
        ShapeLibrary.load(library_path)
        if library_path
        else ShapeLibrary([sphere(resolution=resolution, samples=samples)])
    )
    radius = max(s.radius for s in library.shapes)
    side = int(np.ceil(particles ** (1 / 3)))
    spacing = 1.99 * radius
    indices = np.stack(
        np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), -1
    ).reshape(-1, 3)[:particles]
    positions = (indices + 0.5) * spacing
    cell = np.full(3, side * spacing)
    ids = np.arange(particles, dtype=np.uint32) % len(library.shapes)
    rotations = None
    separations = None
    if packing == "controlled-pairs":
        positions, cell, ids, rotations, separations = controlled_pairs(
            library, particles
        )
    if pairs_per_particle is None:
        pairs_per_particle = 1 if packing == "controlled-pairs" else 8
    predicted_contacts = None
    if packing == "controlled-pairs":
        per_shape = []
        for shape, distance in zip(library.shapes, separations):
            branch = np.array([distance, 0, 0])
            per_shape.append(
                int(
                    np.count_nonzero(shape.query(shape.samples - branch)[0] < 0)
                    + np.count_nonzero(shape.query(shape.samples + branch)[0] < 0)
                )
            )
        predicted_contacts = int(np.array(per_shape)[ids[::2]].sum())
    if contacts_per_particle is None:
        contacts_per_particle = (
            max(16, int(np.ceil(2 * predicted_contacts / particles)))
            if predicted_contacts is not None
            else 256
        )
    spec = Experiment(
        particles=particles,
        contact_geometry=contact_geometry,
        timestep=timestep,
        pairs_per_particle=pairs_per_particle,
        contacts_per_particle=contacts_per_particle,
        max_overlap_fraction=0.01,
    )
    e = Engine(library, spec, positions, cell, ids, np.ones(particles), metal=metal)
    if rotations is not None:
        e.buffers["rotation"].copy_from_numpy(rotations)
    e.initialize()
    before = e.diagnostics()
    committed = e.state() if packing == "controlled-pairs" else None
    e.advance(2)
    durations = []
    diagnostics = []
    study_start = time.perf_counter()
    while (
        len(durations) < repeats or time.perf_counter() - study_start < minimum_seconds
    ):
        if committed is not None:
            e.restore(committed, 0)
        start = time.perf_counter()
        e.advance(steps)
        durations.append((time.perf_counter() - start) / steps)
        diagnostics.append(e.diagnostics())
        if (
            committed is not None
            and diagnostics[-1]["active_grain_pairs"] != particles // 2
        ):
            raise RuntimeError(
                "Controlled pair workload lost contacts or gained cross-pair contacts"
            )
        if len(durations) % 16 == 0:
            write_json(
                output / "progress.json",
                dict(
                    status="running",
                    completed_repeats=len(durations),
                    elapsed_wall_seconds=time.perf_counter() - study_start,
                    latest_seconds_per_step=durations[-1],
                ),
            )
    profile = e.profile_step()
    report = dict(
        format="sand-dem-benchmark/1",
        backend=e.m.doctor(),
        source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        shader_hashes={
            p.name: sha256(p)
            for p in (Path(__file__).parent / "shaders").glob("*.slang")
        },
        workload=dict(
            particles=particles,
            unique_shapes=len(library.shapes),
            geometry="analytical-sphere" if not library_path else "mesh-derived",
            resolution=resolution if not library_path else None,
            samples=samples
            if not library_path
            else [len(s.samples) for s in library.shapes],
            packing=packing,
            contact_geometry=contact_geometry,
            arrangement="isolated contacting pairs; repeated from identical committed state"
            if committed is not None
            else "periodic simple cubic, bounding sphere spacing 1.99r",
            controlled_pair_separations_m=separations,
            timestep_s=timestep,
            pairs_per_particle=pairs_per_particle,
            contacts_per_particle=contacts_per_particle,
            predicted_active_surface_contacts=predicted_contacts,
            capacity_policy="twice preprocessed active contact count, minimum 16 per particle"
            if committed is not None
            else "explicit fixed capacity",
            minimum_wall_seconds=minimum_seconds,
            reset_time_included=False,
            shape_library_sha256=sha256(library_path) if library_path else None,
            steps_per_repeat=steps,
            requested_repeats=repeats,
            repeats=len(durations),
        ),
        seconds_per_step=durations,
        median_seconds_per_step=statistics.median(durations),
        steps_per_second=1 / statistics.median(durations),
        estimated_working_memory_bytes=e.estimated_bytes,
        memory_accounting=memory_report(e),
        elapsed_wall_seconds=time.perf_counter() - study_start,
        first_quarter_median_seconds_per_step=statistics.median(
            durations[: max(1, len(durations) // 4)]
        ),
        last_quarter_median_seconds_per_step=statistics.median(
            durations[-max(1, len(durations) // 4) :]
        ),
        initial_diagnostics=before,
        final_diagnostics=diagnostics[-1],
        gpu_profile=profile,
        scientific_verification=False,
        limitations=[
            "This workload is not a prepared soil specimen. Minimum duration is explicit; a long run alone does not prove thermal equilibrium.",
            "Wall time includes Python command encoding, dispatch and scalar boundary reads.",
            "Controlled pairs exercise irregular contacts but do not establish dense assembly performance.",
            "Reset and diagnostic costs are excluded from per-step timing; elapsed wall time includes them.",
        ],
    )
    write_json(output / "benchmark.json", report)
    return report
