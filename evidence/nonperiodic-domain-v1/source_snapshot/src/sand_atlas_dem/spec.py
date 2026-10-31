"""Versioned, strict experiment contract. Public values use SI units."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .util import finite_array, read_json


@dataclass
class Experiment:
    format: str = "sand-dem-experiment/1"
    shape_library: str = ""
    contact_geometry: str = "sdf"
    periodic: bool = True
    particles: int = 1000
    seed: int = 42
    density: float = 2650.0
    normal_stiffness: float = 1e12  # traction/penetration: Pa/m
    tangential_stiffness: float = 5e11
    friction: float = 0.5
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
        if type(self.periodic) is not bool:
            raise ValueError("periodic must be an explicit boolean")
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
        }
        for s in self.stages:
            if not self.periodic and s.get("kind") != "hold":
                raise ValueError(
                    "Nonperiodic domains currently support hold stages only"
                )
            if set(s) - allowed or s.get("kind") not in {
                "hold",
                "isotropic",
                "triaxial",
                "strain",
            }:
                raise ValueError("Unsupported stage or unknown stage fields")
            if type(s.get("steps")) is not int or s["steps"] < 1:
                raise ValueError("Positive stage steps required")
            if not np.isfinite(s.get("drag", 0)) or s.get("drag", 0) < 0:
                raise ValueError("Drag must be finite and nonnegative")
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
