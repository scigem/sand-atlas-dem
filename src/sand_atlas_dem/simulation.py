"""Periodic virtual laboratory, reproducible bundles and strict restarts."""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .geometry import ShapeLibrary
from .gpu import Engine
from .spec import Experiment
from .util import read_json, sha256, write_json


@dataclass
class RunResult:
    path: Path
    status: str
    rows: list


def initial_assembly(library, spec):
    rng = np.random.default_rng(spec.seed)
    n = spec.particles
    ids = rng.integers(len(library.shapes), size=n, dtype=np.uint32)
    scales = np.full(n, spec.scale)
    radius = max(s.radius for s in library.shapes) * spec.scale
    side = int(np.ceil(n ** (1 / 3)))
    vol = sum(library.shapes[int(i)].volume * spec.scale**3 for i in ids)
    length = (vol / spec.initial_solid_fraction) ** (1 / 3)
    cell = np.full(3, length) if spec.cell is None else np.asarray(spec.cell, float)
    spacing = cell / side
    if np.any(spacing <= 2.1 * radius):
        raise ValueError(
            "Requested dilute assembly does not fit nonoverlapping bounding spheres; reduce initial_solid_fraction or enlarge cell"
        )
    indices = np.stack(
        np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), -1
    ).reshape(-1, 3)
    indices = indices[rng.permutation(len(indices))[:n]]
    positions = (indices + 0.5) * spacing
    jitter = np.maximum(0, spacing - 2.1 * radius) * 0.1
    positions += rng.uniform(-1, 1, (n, 3)) * jitter
    # Positions are provably nonoverlapping even with jitter: final spacing > 2r.
    return positions, cell, ids, scales


class Simulation:
    def __init__(self, experiment, *, library=None, metal=None):
        self.experiment = experiment
        self.library = library or ShapeLibrary.load(experiment.shape_library)
        positions, cell, ids, scales = initial_assembly(self.library, experiment)
        self.engine = Engine(
            self.library, experiment, positions, cell, ids, scales, metal=metal
        )
        rng = np.random.default_rng(experiment.seed + 1)
        q = rng.normal(size=(experiment.particles, 4))
        q /= np.linalg.norm(q, axis=1)[:, None]
        self.engine.buffers["rotation"].copy_from_numpy(q.astype(np.float32))
        self.engine.initialize()
        self.rows = []
        self.stage_index = 0
        self.stage_steps = 0
        self.filter_stress = np.zeros(3)
        self.converged = 0
        self.contact_phase = False

    def checkpoint(self, path):
        state = self.engine.state()
        with h5py.File(path, "w") as f:
            f.attrs.update(
                format="sand-dem-checkpoint/3",
                step=self.engine.step,
                stage_index=self.stage_index,
                stage_steps=self.stage_steps,
                converged=self.converged,
                contact_phase=self.contact_phase,
                experiment=json.dumps(self.engine.spec.to_dict()),
                shape_sha256=sha256(self.experiment.shape_library),
                filter_stress=json.dumps(self.filter_stress.tolist()),
                initial_cell=json.dumps(self.engine.initial_cell.tolist()),
                rows=json.dumps(self.rows),
            )
            for name, value in state.items():
                if name == "contacts":
                    f.attrs[name] = value
                else:
                    f.create_dataset(name, data=value, compression="gzip")

    @classmethod
    def resume(cls, path, *, metal=None):
        with h5py.File(path) as f:
            if f.attrs.get("format") != "sand-dem-checkpoint/3":
                raise ValueError("Unsupported checkpoint")
            spec = Experiment(**json.loads(f.attrs["experiment"]))
            if sha256(spec.shape_library) != f.attrs["shape_sha256"]:
                raise ValueError("Shape library changed since checkpoint")
            state = {name: f[name][:] for name in f}
            state["contacts"] = int(f.attrs["contacts"])
            obj = cls(spec, metal=metal)
            obj.engine.initial_cell = np.array(json.loads(f.attrs["initial_cell"]))
            obj.engine.restore(state, int(f.attrs["step"]))
            obj.stage_index = int(f.attrs["stage_index"])
            obj.stage_steps = int(f.attrs["stage_steps"])
            obj.converged = int(f.attrs["converged"])
            obj.contact_phase = bool(f.attrs.get("contact_phase", False))
            obj.filter_stress = np.array(json.loads(f.attrs["filter_stress"]))
            obj.rows = json.loads(f.attrs["rows"])
        return obj

    def rate(self, stage, row):
        kind = stage["kind"]
        dt = self.experiment.diagnostic_interval * self.experiment.timestep
        if kind == "hold":
            return np.zeros(3)
        if kind == "strain":
            return np.asarray(stage["rate"], float)
        diagonal = np.diag(row["stress"])
        alpha = -np.expm1(-dt / stage["filter_time"])
        self.filter_stress += (diagonal - self.filter_stress) * alpha
        self.contact_phase = self.contact_phase or row["active_surface_contacts"] > 0
        max_rate = stage["max_rate"]
        if self.contact_phase:
            max_rate = min(max_rate, stage.get("contact_max_rate", max_rate))
        rate = np.clip(
            stage["gain"]
            * (self.filter_stress - stage["pressure"])
            / stage["pressure"],
            -max_rate,
            max_rate,
        )
        if kind == "triaxial":
            rate[2] = -stage["rate"]
        return rate

    def run(self, output, *, max_chunks=None):
        path = Path(output)
        path.mkdir(parents=True, exist_ok=False)
        spec = self.experiment
        start = time.perf_counter()
        status = "running"
        failure = None
        chunks = 0
        manifest = dict(
            format="sand-dem-run/1",
            kind="synthetic-dem",
            scientific_admission=False,
            numerics=dict(
                state_accumulation="compensated_fp32",
                cell_update="accumulated_log_strain",
                checkpoint_format="sand-dem-checkpoint/3",
            ),
            experiment=spec.to_dict(),
            shape_sha256=sha256(spec.shape_library),
            backend=self.engine.m.doctor(),
            estimated_memory_bytes=self.engine.estimated_bytes,
            conventions=dict(
                stress="compression_positive",
                tensor_order="xx,xy,xz;yx,yy,yz;zx,zy,zz",
                stress_measure="contact_virial_plus_peculiar_kinetic",
                strain="engineering_compression_positive",
                volumetric_strain="sum_of_axial_engineering_strains",
                void_ratio="cell_volume_over_sum_rigid_grain_volumes_minus_one",
            ),
            units=dict(length="m", stress="Pa", time="s", strain="1"),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_hashes={
                p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
            },
            shader_hashes={
                p.name: sha256(p)
                for p in (Path(__file__).parent / "shaders").glob("*.slang")
            },
        )
        write_json(path / "manifest.json", manifest)
        if not self.rows:
            self.rows.append(dict(self.engine.diagnostics(), stage=self.stage_index))
        try:
            while self.stage_index < len(spec.stages):
                stage = spec.stages[self.stage_index]
                if self.stage_steps >= stage["steps"]:
                    if stage["kind"] == "isotropic":
                        raise RuntimeError(
                            "Isotropic preparation exhausted its step budget before convergence"
                        )
                    self.stage_index += 1
                    self.stage_steps = 0
                    self.converged = 0
                    self.contact_phase = False
                    if self.stage_index >= len(spec.stages):
                        break
                    stage = spec.stages[self.stage_index]
                    self.engine.initial_cell = np.asarray(self.rows[-1]["cell"])
                amount = min(
                    spec.diagnostic_interval, stage["steps"] - self.stage_steps
                )
                rate = self.rate(stage, self.rows[-1])
                wall = time.perf_counter()
                self.engine.advance(amount, rate=rate, drag=stage.get("drag", 0))
                self.stage_steps += amount
                chunks += 1
                row = dict(
                    self.engine.diagnostics(),
                    stage=self.stage_index,
                    wall_seconds=time.perf_counter() - wall,
                )
                origin = self.rows[0]
                row["work_balance_proxy_residual"] = (
                    row["cell_work"]
                    - origin["cell_work"]
                    - row["friction_dissipation"]
                    + origin["friction_dissipation"]
                    - row["drag_dissipation"]
                    + origin["drag_dissipation"]
                    - row["kinetic_energy"]
                    + origin["kinetic_energy"]
                    - row["elastic_energy_proxy"]
                    + origin["elastic_energy_proxy"]
                )
                self.rows.append(row)
                if self.engine.step % spec.snapshot_interval == 0:
                    self.snapshot(path / f"particles-{self.engine.step:09d}.h5")
                if stage["kind"] == "isotropic":
                    pressure = np.max(
                        np.abs(np.diag(row["stress"]) / stage["pressure"] - 1)
                    )
                    ratio = row["kinetic_energy"] / max(
                        row["elastic_energy_proxy"], 1e-30
                    )
                    ok = (
                        pressure <= stage.get("pressure_tolerance", 0.01)
                        and ratio <= stage.get("kinetic_tolerance", 1e-5)
                        and row["unbalanced_force"]
                        <= stage.get("force_tolerance", 1e-3)
                    )
                    self.converged = self.converged + 1 if ok else 0
                    if self.converged >= stage.get("converged_samples", 100):
                        prepared_stage = self.stage_index
                        self.stage_index += 1
                        self.stage_steps = 0
                        self.converged = 0
                        self.contact_phase = False
                        self.engine.initial_cell = np.asarray(row["cell"])
                        self.checkpoint(path / f"prepared-stage-{prepared_stage}.h5")
                if stage["kind"] == "triaxial" and row["axial_strain"] >= stage.get(
                    "stop_strain", 0.1
                ):
                    self.stage_index += 1
                    self.stage_steps = 0
                if max_chunks is not None and chunks >= max_chunks:
                    status = "paused"
                    break
            else:
                status = "complete"
            if self.stage_index >= len(spec.stages):
                status = "complete"
        except Exception as exc:
            status = "failed"
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            write_json(
                path / "history.json",
                dict(format="sand-dem-history/1", kind="synthetic-dem", rows=self.rows),
            )
            if not getattr(self.engine, "invalid", False):
                self.checkpoint(path / "checkpoint.h5")
            manifest.update(
                status=status,
                failure=failure,
                elapsed_wall_seconds=time.perf_counter() - start,
                completed_steps=self.engine.step,
                stage_index=self.stage_index,
                capacity_retries=self.engine.capacity_retries,
            )
            manifest["artifacts"] = {
                p.name: sha256(p)
                for p in path.iterdir()
                if p.is_file() and p.name != "manifest.json"
            }
            write_json(path / "manifest.json", manifest)
        return RunResult(path, status, self.rows)

    def snapshot(self, path):
        state = self.engine.state()
        with h5py.File(path, "w") as f:
            f.attrs.update(
                format="sand-dem-snapshot/2",
                step=self.engine.step,
                length_unit_m=self.engine.length_unit,
            )
            for name in (
                "position",
                "rotation",
                "cell",
                "positionRemainder",
                "rotationRemainder",
            ):
                f.create_dataset(name, data=state[name])
            f.create_dataset("shape_id", data=self.engine.ids)
            f.create_dataset("scale", data=self.engine.scales)


def export_run(path, output):
    path = Path(path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = read_json(path / "manifest.json")
    history = read_json(path / "history.json")
    if sha256(manifest["experiment"]["shape_library"]) != manifest["shape_sha256"]:
        raise ValueError("Shape library changed since run")
    for name, digest in manifest["artifacts"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Run artifact hash mismatch: {name}")
    channels = [
        "step",
        "elapsed_time",
        "stage",
        "axial_strain",
        "volumetric_strain",
        "volume_ratio",
        "p",
        "q",
        "void_ratio",
        "kinetic_energy",
        "elastic_energy_proxy",
        "unbalanced_force",
        "max_overlap_fraction",
        "candidate_pairs",
        "active_surface_contacts",
        "active_grain_pairs",
        "coordination",
        "sliding_fraction",
        "friction_dissipation",
        "drag_dissipation",
        "cell_work",
        "work_balance_proxy_residual",
    ]
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, channels, extrasaction="ignore")
        w.writeheader()
        w.writerows(history["rows"])
    write_json(
        out / "atlas-comparison.json",
        dict(
            format="sand-dem-atlas-comparison/1",
            kind="synthetic-dem",
            run_status=manifest["status"],
            scientific_admission=False,
            protocol="drained-triaxial"
            if any(s["kind"] == "triaxial" for s in manifest["experiment"]["stages"])
            else "research",
            source_manifest_sha256=sha256(path / "manifest.json"),
            conventions=manifest["conventions"],
            units=manifest["units"],
            preparation=manifest["experiment"]["stages"],
            warnings=[
                "Synthetic DEM evidence, not measured experimental validation.",
                "Finite-strain histories are not automatically compatible with infinitesimal continuum models.",
                "Nominal void ratio ignores geometric overlap volume.",
            ],
            initial=history["rows"][0],
            rows=history["rows"],
        ),
    )
    return out
