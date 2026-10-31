"""Command-line entry point. GPU commands fail rather than fall back to CPU."""

import argparse
import json
import sys
from pathlib import Path

from .util import sha256, write_json


def _view_arguments(parser):
    from .viewer import FIELDS, VECTORS

    parser.add_argument(
        "--live", action="store_true", help="Interactive 3D viewer during simulation"
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Write x/y/z/isometric MP4s under the run's visualization directory",
    )
    parser.add_argument(
        "--view-every", type=int, default=1, help="Render every N diagnostic chunks"
    )
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--color-by", choices=FIELDS, default="speed")
    parser.add_argument("--vectors", choices=VECTORS, default="none")
    parser.add_argument(
        "--color-max", type=float, help="Fixed colour scale maximum in SI units"
    )
    parser.add_argument("--max-vectors", type=int, default=1000)
    parser.add_argument("--view-geometry", choices=["mesh", "sphere"], default="mesh")


def _view_options(args):
    from .viewer import ViewOptions

    if not (args.live or args.record_video):
        return None
    options = ViewOptions(
        args.live,
        args.record_video,
        args.view_every,
        args.video_fps,
        args.color_by,
        args.vectors,
        args.color_max,
        args.max_vectors,
        args.view_geometry,
    )
    options.preflight()
    return options


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
    a.add_argument("--atlas-root", type=Path)
    a.add_argument("--atlas-policy", type=Path)
    a.add_argument("--atlas-config", type=Path)
    _view_arguments(a)
    atlas = sub.add_parser("atlas").add_subparsers(dest="atlas_command", required=True)
    a = atlas.add_parser("prepare")
    a.add_argument("run", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--policy", type=Path, required=True)
    a.add_argument("--configuration", type=Path, required=True)
    a.add_argument("--atlas-root", type=Path, required=True)
    a = atlas.add_parser(
        "compare",
        help="Replay a synthetic handoff through an existing Atlas native solver",
    )
    a.add_argument("handoff", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--atlas-root", type=Path, required=True)
    a.add_argument("--solver", type=Path, required=True)
    a.add_argument("--catalogue", type=Path, required=True)
    a.add_argument(
        "--plots",
        action="store_true",
        help="Write an SVG response comparison (requires inspect extra)",
    )
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
    a.add_argument("--subdivisions", type=int, default=0)
    a = g.add_parser("adaptive-quadrature")
    a.add_argument("library", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--reference", type=Path, required=True)
    a.add_argument("--reference-protocol", type=Path, required=True)
    a.add_argument("--target-error", type=float, default=0.05)
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
    _view_arguments(a)
    a = sub.add_parser("resume")
    a.add_argument("checkpoint", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--max-chunks", type=int)
    _view_arguments(a)
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

            handoff = (args.atlas_root, args.atlas_policy, args.atlas_config)
            if any(handoff) and not all(handoff):
                raise ValueError(
                    "Supply --atlas-root, --atlas-policy and --atlas-config together"
                )
            if all(handoff):
                read_paths = [
                    args.atlas_policy,
                    args.atlas_config,
                    args.atlas_root / "atlas/experiment_replay.py",
                ]
                if any(not path.is_file() for path in read_paths):
                    raise ValueError(
                        "Atlas configuration, policy and prepare-replay checkout must exist"
                    )
                from .atlas_bridge import _snapshot, validate_policy

                policy, policy_hash = _snapshot(args.atlas_policy)
                configuration, configuration_hash = _snapshot(args.atlas_config)
                validate_policy(policy)
                frozen_handoff = dict(
                    policy=policy,
                    configuration=configuration,
                    policy_source_sha256=policy_hash,
                    configuration_source_sha256=configuration_hash,
                    atlas_root=str(args.atlas_root.resolve()),
                )
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
            if all(handoff):
                report["atlas_handoff"] = frozen_handoff
            if not args.dry_run:
                if args.output is None:
                    raise ValueError(
                        "--output is required unless --dry-run is specified"
                    )
                view_options = _view_options(args)
                view_kwargs = (
                    {"view_options": view_options} if view_options is not None else {}
                )
                report = run_case(report, args.output, args.max_chunks, **view_kwargs)
                if all(handoff) and report["status"] == "complete":
                    from .atlas_bridge import export_replay

                    try:
                        report["atlas"] = export_replay(
                            args.output / "run",
                            args.output / "atlas",
                            policy_path=args.output / "atlas-policy.json",
                            configuration=args.output / "atlas-configuration.json",
                            atlas_root=args.atlas_root,
                        )
                    except Exception as exc:
                        report["atlas"] = dict(
                            status="failed", error=f"{type(exc).__name__}: {exc}"
                        )
                        write_json(args.output / "result.json", report)
                        raise
                    write_json(args.output / "result.json", report)
                print(json.dumps(report, indent=2, allow_nan=False))
                return 1 if report["status"] == "failed" else 0
        elif args.command == "atlas":
            if args.atlas_command == "compare":
                from .atlas_compare import replay_comparison

                report = replay_comparison(
                    args.handoff,
                    args.output,
                    atlas_root=args.atlas_root,
                    solver=args.solver,
                    catalogue=args.catalogue,
                    plots=args.plots,
                )
            else:
                from .atlas_bridge import export_replay

                report = export_replay(
                    args.run,
                    args.output,
                    policy_path=args.policy,
                    configuration=args.configuration,
                    atlas_root=args.atlas_root,
                )
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
                    with_face_quadrature(shape, args.rule, args.subdivisions)
                    for shape in library.shapes
                ]
                for shape in shapes:
                    shape.provenance["parent_shape_library_sha256"] = parent_hash
                ShapeLibrary(shapes).save(args.output)
                report = dict(
                    output=str(args.output),
                    sha256=sha256(args.output),
                    parent_sha256=parent_hash,
                    rule=args.rule,
                    subdivisions=args.subdivisions,
                    samples=[len(shape.samples) for shape in shapes],
                )
            elif args.geometry_command == "adaptive-quadrature":
                from .quadrature import prepare_adaptive

                report = prepare_adaptive(
                    args.library,
                    args.reference,
                    args.reference_protocol,
                    args.output,
                    target=args.target_error,
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

            view_options = _view_options(args)
            sim = (
                Simulation(Experiment.load(args.specification))
                if args.command == "run"
                else Simulation.resume(args.checkpoint)
            )
            result = sim.run(
                args.output, max_chunks=args.max_chunks, view_options=view_options
            )
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
