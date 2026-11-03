"""Versioned, strict experiment contract. Public values use SI units."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .boundaries import validate_planes
from .util import finite_array, read_json


@dataclass
class Experiment:
    format: str = "sand-dem-experiment/1"
    shape_library: str = ""
    contact_geometry: str = "sdf"
    contact_normal: str = "target-gradient"
    periodic: bool = True
    planes: list = field(default_factory=list)
    boundary_box: list | None = None
    particles: int = 1000
    shape_sampling: str = "random"
    seed: int = 42
    density: float = 2650.0
    normal_stiffness: float = 1e12  # traction/penetration: Pa/m
    tangential_stiffness: float = 5e11
    friction: float = 0.5
    normal_viscosity: float = 0.0  # normal traction / normal speed: N*s/m^3
    timestep_policy: str = "conservative"
    timestep: float = 1e-7
    initial_solid_fraction: float = 0.15
    scale: float = 1.0
    cell: list | None = None
    skin: float = 0.05  # fraction of maximum particle diameter
    working_memory_gib: float = 12.0
    contacts_per_particle: int = 256
    pairs_per_particle: int = 32
    bin_capacity: int = 128
    stages: list = field(
        default_factory=lambda: [{"kind": "hold", "steps": 1000, "drag": 0.0}]
    )
    diagnostic_interval: int = 50
    snapshot_interval: int = 1000
    max_overlap_fraction: float = 0.01

    def __post_init__(self):
        if self.format != "sand-dem-experiment/1":
            raise ValueError("Unsupported experiment format")
        if self.contact_geometry not in {"sdf", "mesh"}:
            raise ValueError("Unsupported contact geometry")
        if self.contact_normal not in {"target-gradient", "source-surface-experimental"}:
            raise ValueError("Unsupported contact normal")
        if self.contact_normal == "source-surface-experimental" and (
            self.contact_geometry != "mesh" or self.friction != 0
            or self.normal_viscosity != 0 or self.planes
        ):
            raise ValueError("Experimental source normals require mesh, zero friction/viscosity and no planes")
        if self.timestep_policy not in {"conservative", "active-contact"}:
            raise ValueError("Unsupported timestep policy")
        if self.shape_sampling not in {"random", "balanced"}:
            raise ValueError("shape_sampling must be random or balanced")
        if type(self.periodic) is not bool:
            raise ValueError("periodic must be an explicit boolean")
        validate_planes(self.planes, self.periodic, self.boundary_box)
        for name in (
            "particles",
            "seed",
            "contacts_per_particle",
            "pairs_per_particle",
            "bin_capacity",
            "diagnostic_interval",
            "snapshot_interval",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"Invalid integer {name}")
        for name in (
            "density",
            "normal_stiffness",
            "tangential_stiffness",
            "timestep",
            "scale",
            "skin",
            "working_memory_gib",
            "max_overlap_fraction",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid positive {name}")
        if (
            not np.isfinite(self.friction)
            or self.friction < 0
            or not np.isfinite(self.normal_viscosity)
            or self.normal_viscosity < 0
            or not 0 < self.initial_solid_fraction < 0.5
        ):
            raise ValueError("Invalid friction or initial solid fraction")
        if self.cell is not None:
            finite_array(self.cell, (3,), positive=True)
        if not self.stages:
            raise ValueError("At least one stage required")
        allowed = {
            "kind",
            "steps",
            "drag",
            "rate",
            "pressure",
            "gain",
            "filter_time",
            "max_rate",
            "contact_max_rate",
            "converged_samples",
            "stop_strain",
            "pressure_tolerance",
            "kinetic_tolerance",
            "force_tolerance",
            "wall_velocities",
        }
        for s in self.stages:
            if not self.periodic and s.get("kind") not in {"hold", "wall"}:
                raise ValueError("Nonperiodic domains support hold or wall stages only")
            if set(s) - allowed or s.get("kind") not in {
                "hold",
                "isotropic",
                "triaxial",
                "strain",
                "wall",
            }:
                raise ValueError("Unsupported stage or unknown stage fields")
            if type(s.get("steps")) is not int or s["steps"] < 1:
                raise ValueError("Positive stage steps required")
            if not np.isfinite(s.get("drag", 0)) or s.get("drag", 0) < 0:
                raise ValueError("Drag must be finite and nonnegative")
            if s["kind"] == "wall":
                if not self.planes:
                    raise ValueError("Wall stages require planes")
                finite_array(s.get("wall_velocities"), (len(self.planes), 3))
            elif "wall_velocities" in s:
                raise ValueError("wall_velocities require a wall stage")
            if s["kind"] in {"isotropic", "triaxial"}:
                for key in ("pressure", "gain", "filter_time", "max_rate"):
                    if not np.isfinite(s.get(key, np.nan)) or s[key] <= 0:
                        raise ValueError(f"Explicit positive servo {key} required")
            for key in (
                "pressure_tolerance",
                "kinetic_tolerance",
                "force_tolerance",
                "stop_strain",
                "contact_max_rate",
            ):
                if key in s and (not np.isfinite(s[key]) or s[key] <= 0):
                    raise ValueError(f"Positive finite {key} required")
            if "converged_samples" in s and (
                type(s["converged_samples"]) is not int or s["converged_samples"] < 1
            ):
                raise ValueError("Positive integer converged_samples required")
            if s["kind"] == "strain":
                finite_array(s.get("rate"), (3,))
            if s["kind"] == "triaxial" and (
                not np.isfinite(s.get("rate", np.nan)) or s["rate"] <= 0
            ):
                raise ValueError("Triaxial compression rate must be positive")

    @classmethod
    def load(cls, path):
        value = read_json(path)
        value["shape_library"] = str(
            (Path(path).parent / value["shape_library"]).resolve()
        )
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    @property
    def viscous(self):
        return self.normal_viscosity > 0 or any(
            p.get("normal_viscosity", 0) > 0 for p in self.planes
        )

    @property
    def checkpoint_format(self):
        return (
            "sand-dem-checkpoint/5"
            if self.viscous
            else "sand-dem-checkpoint/4"
            if self.planes
            else "sand-dem-checkpoint/3"
        )
