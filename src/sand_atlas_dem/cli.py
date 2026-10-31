"""Command-line entry point. GPU commands fail rather than fall back to CPU."""

import argparse
import json
import sys
from pathlib import Path

from .util import sha256, write_json


def main(argv=None):
    p = argparse.ArgumentParser(prog="sand-dem")
    sub = p.add_subparsers(dest="command", required=True)
    cache_default = Path.home() / ".cache" / "sand-atlas-dem" / "materials"
    material = sub.add_parser("material").add_subparsers(
        dest="material_command", required=True
    )
    a = material.add_parser("list")
    a.add_argument("--cache", type=Path, default=cache_default)
    a = material.add_parser("prepare")
    a.add_argument("material")
    a.add_argument("--cache", type=Path, default=cache_default)
    a.add_argument("--resolution", type=int, default=64)
    a.add_argument("--samples", type=int, default=2048)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--source-directory", type=Path)
    a = material.add_parser("register")
    a.add_argument("receipt", type=Path)
    a.add_argument("--cache", type=Path, default=cache_default)
    case = sub.add_parser("case").add_subparsers(dest="case_command", required=True)
    a = case.add_parser("run")
    a.add_argument("material")
    a.add_argument("--case", type=Path, required=True)
    a.add_argument("--cache", type=Path, default=cache_default)
    a.add_argument("--output", type=Path)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--max-chunks", type=int)
    a.add_argument("--particles", type=int)
    a.add_argument("--seed", type=int)
    a.add_argument(
        "--pressure", type=float, help="Pa; overrides all pressure-controlled stages"
    )
    a.add_argument("--rate", type=float, help="positive axial compression rate, 1/s")
    a.add_argument("--stop-strain", type=float)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--output", type=Path)
    g = sub.add_parser("geometry").add_subparsers(
        dest="geometry_command", required=True
    )
    a = g.add_parser("prepare")
    a.add_argument("mesh", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--unit", choices=["m", "mm", "um"], required=True)
    a.add_argument("--resolution", type=int, default=64)
    a.add_argument("--samples", type=int, default=2048)
    a.add_argument("--seed", type=int, default=0)
    a = g.add_parser("face-quadrature")
    a.add_argument("library", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--rule", choices=["centroid", "gauss3"], default="gauss3")
    a = g.add_parser("inspect")
    a.add_argument("library", type=Path)
    a.add_argument("--output", type=Path)
    a = g.add_parser("sphere")
    a.add_argument("output", type=Path)
    a.add_argument("--radius", type=float, default=0.0005)
    a.add_argument("--resolution", type=int, default=32)
    a.add_argument("--samples", type=int, default=512)
    a = sub.add_parser("run")
    a.add_argument("specification", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--max-chunks", type=int)
    a = sub.add_parser("resume")
    a.add_argument("checkpoint", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--max-chunks", type=int)
    a = sub.add_parser("benchmark")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--particles", type=int, default=1000)
    a.add_argument("--resolution", type=int, default=32)
    a.add_argument("--samples", type=int, default=512)
    a.add_argument("--steps", type=int, default=10)
    a.add_argument("--repeats", type=int, default=3)
    a.add_argument("--library", type=Path)
    a.add_argument(
        "--packing",
        choices=["simple-cubic", "controlled-pairs"],
        default="simple-cubic",
    )
    a.add_argument("--contact-geometry", choices=["sdf", "mesh"], default="sdf")
    a.add_argument("--timestep", type=float, default=1e-8)
    a.add_argument("--minimum-seconds", type=float, default=0)
    a.add_argument("--pairs-per-particle", type=int)
    a.add_argument("--contacts-per-particle", type=int)
    a = sub.add_parser("qualify")
    a.add_argument("protocol", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("export")
    a.add_argument("run", type=Path)
    a.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "material":
            from .cases import materials, register_material

            if args.material_command == "prepare":
                from .acquisition import prepare_material

                report = prepare_material(
                    args.material,
                    args.cache,
                    resolution=args.resolution,
                    samples=args.samples,
                    workers=args.workers,
                    source_directory=args.source_directory,
                )
            else:
                report = (
                    materials(args.cache)
                    if args.material_command == "list"
                    else register_material(args.receipt, args.cache)
                )
        elif args.command == "case":
            from .cases import MaterialNotFound, resolve_case, run_case

            overrides = {
                key: getattr(args, key)
                for key in ("particles", "seed", "pressure", "rate", "stop_strain")
            }
            if not args.dry_run and args.output is None:
                raise ValueError("--output is required unless --dry-run is specified")
            if not args.dry_run and args.output.exists():
                raise FileExistsError(args.output)
            try:
                report = resolve_case(args.material, args.case, args.cache, overrides)
            except MaterialNotFound:
                if args.dry_run:
                    raise
                from .acquisition import prepare_material

                print(
                    "Preparing reviewed material geometry (CPU); cached sources remain private.",
                    file=sys.stderr,
                )
                prepare_material(args.material, args.cache)
                report = resolve_case(args.material, args.case, args.cache, overrides)
            if not args.dry_run:
                if args.output is None:
                    raise ValueError(
                        "--output is required unless --dry-run is specified"
                    )
                report = run_case(report, args.output, args.max_chunks)
                print(json.dumps(report, indent=2, allow_nan=False))
                return 1 if report["status"] == "failed" else 0
        elif args.command == "doctor":
            from .backend import Metal

            report = Metal().doctor()
            if args.output:
                write_json(args.output, report)
        elif args.command == "geometry":
            from .geometry import ShapeLibrary, prepare, sphere

            if args.geometry_command == "prepare":
                s = prepare(
                    args.mesh,
                    args.output,
                    unit=args.unit,
                    resolution=args.resolution,
                    samples=args.samples,
                    seed=args.seed,
                )
                report = dict(
                    id=s.id,
                    volume_m3=s.volume,
                    samples=len(s.samples),
                    grid=list(s.sdf.shape),
                )
            elif args.geometry_command == "face-quadrature":
                from .geometry import with_face_quadrature

                if args.output.exists():
                    raise FileExistsError(args.output)
                parent_hash = sha256(args.library)
                library = ShapeLibrary.load(args.library)
                shapes = [
                    with_face_quadrature(shape, args.rule) for shape in library.shapes
                ]
                for shape in shapes:
                    shape.provenance["parent_shape_library_sha256"] = parent_hash
                ShapeLibrary(shapes).save(args.output)
                report = dict(
                    output=str(args.output),
                    sha256=sha256(args.output),
                    parent_sha256=parent_hash,
                    rule=args.rule,
                    samples=[len(shape.samples) for shape in shapes],
                )
            elif args.geometry_command == "sphere":
                ShapeLibrary([sphere(args.radius, args.resolution, args.samples)]).save(
                    args.output
                )
                report = dict(output=str(args.output), fixture="analytical sphere")
            else:
                lib = ShapeLibrary.load(args.library)
                report = dict(
                    shapes=[
                        dict(
                            id=s.id,
                            grid=list(s.sdf.shape),
                            spacing_m=s.spacing,
                            volume_m3=s.volume,
                            inertia_unit_density=s.inertia.tolist(),
                            sample_area_m2=float(s.areas.sum()),
                            provenance=s.provenance,
                        )
                        for s in lib.shapes
                    ]
                )
                if args.output:
                    from .visualize import inspect_shapes

                    inspect_shapes(lib, args.output)
        elif args.command in {"run", "resume"}:
            from .simulation import Simulation
            from .spec import Experiment

            sim = (
                Simulation(Experiment.load(args.specification))
                if args.command == "run"
                else Simulation.resume(args.checkpoint)
            )
            result = sim.run(args.output, max_chunks=args.max_chunks)
            report = dict(
                path=str(result.path), status=result.status, rows=len(result.rows)
            )
            print(json.dumps(report, indent=2))
            return 1 if result.status == "failed" else 0
        elif args.command == "benchmark":
            from .benchmark import benchmark

            report = benchmark(
                args.output,
                particles=args.particles,
                resolution=args.resolution,
                samples=args.samples,
                steps=args.steps,
                repeats=args.repeats,
                library_path=args.library,
                packing=args.packing,
                contact_geometry=args.contact_geometry,
                timestep=args.timestep,
                minimum_seconds=args.minimum_seconds,
                pairs_per_particle=args.pairs_per_particle,
                contacts_per_particle=args.contacts_per_particle,
            )
        elif args.command == "qualify":
            from .qualification import qualify

            report = qualify(args.protocol, args.output)
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "passed" else 1
        elif args.command == "export":
            from .simulation import export_run
            from .visualize import export_snapshots

            export_run(args.run, args.output)
            export_snapshots(args.run, args.output)
            report = dict(path=str(args.output))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
