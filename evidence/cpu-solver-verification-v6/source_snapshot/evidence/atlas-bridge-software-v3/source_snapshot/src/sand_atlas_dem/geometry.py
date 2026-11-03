"""Verified, explicitly scaled mesh-derived shape assets. All stored lengths are metres."""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import trimesh

from .util import finite_array, sha256

FORMAT = "sand-dem-shapes/1"
UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6}


@dataclass
class Shape:
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    samples: np.ndarray
    areas: np.ndarray
    sdf: np.ndarray
    origin: np.ndarray
    spacing: float
    volume: float
    inertia: np.ndarray  # unit density, kg m² when multiplied by kg/m³
    radius: float
    provenance: dict

    def validate(self):
        for a in (
            self.vertices,
            self.samples,
            self.areas,
            self.sdf,
            self.origin,
            self.inertia,
        ):
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite shape data")
        if (
            self.sdf.ndim != 3
            or min(self.sdf.shape) < 4
            or not (self.sdf.min() < 0 < self.sdf.max())
        ):
            raise ValueError("SDF must be a 3D signed field, negative inside")
        if (
            self.spacing <= 0
            or self.volume <= 0
            or self.radius <= 0
            or np.any(self.areas <= 0)
        ):
            raise ValueError("Invalid shape measure")
        if len(self.samples) != len(self.areas) or self.samples.shape[1:] != (3,):
            raise ValueError("Invalid surface samples")
        if np.linalg.eigvalsh(self.inertia).min() <= 0:
            raise ValueError("Inertia is not positive definite")
        boundary = np.concatenate(
            [
                self.sdf[0].ravel(),
                self.sdf[-1].ravel(),
                self.sdf[:, 0].ravel(),
                self.sdf[:, -1].ravel(),
                self.sdf[:, :, 0].ravel(),
                self.sdf[:, :, -1].ravel(),
            ]
        )
        if np.any(boundary <= 0):
            raise ValueError("SDF exterior padding is not outside the grain")

    def query(self, points):
        """Trilinear interpolation and its exact piecewise gradient, FP64 reference."""
        p = np.atleast_2d(finite_array(points))
        g = (p - self.origin) / self.spacing
        ijk = np.floor(g).astype(int)
        valid = np.all((ijk >= 0) & (ijk < np.array(self.sdf.shape) - 1), axis=1)
        values = np.full(len(p), np.inf)
        gradients = np.zeros((len(p), 3))
        if np.any(valid):
            base = ijk[valid]
            t = g[valid] - base
            v = np.zeros(len(base))
            grad = np.zeros((len(base), 3))
            for i in (0, 1):
                for j in (0, 1):
                    for k in (0, 1):
                        bits = np.array([i, j, k])
                        w = np.where(bits, t, 1 - t)
                        f = self.sdf[tuple((base + bits).T)].astype(float)
                        v += f * w.prod(axis=1)
                        for axis in range(3):
                            other = [d for d in range(3) if d != axis]
                            grad[:, axis] += (
                                f
                                * (2 * bits[axis] - 1)
                                * w[:, other].prod(axis=1)
                                / self.spacing
                            )
            values[valid] = v
            gradients[valid] = grad
        return values, gradients


class ShapeLibrary:
    def __init__(self, shapes):
        self.shapes = list(shapes)
        if not self.shapes or len({s.id for s in self.shapes}) != len(self.shapes):
            raise ValueError("Shape library must contain uniquely identified shapes")
        for shape in self.shapes:
            shape.validate()

    def save(self, path):
        with h5py.File(path, "w") as f:
            f.attrs["format"] = FORMAT
            f.attrs["units"] = "m"
            for index, s in enumerate(self.shapes):
                g = f.create_group(str(index))
                g.attrs.update(
                    id=s.id,
                    spacing=s.spacing,
                    volume=s.volume,
                    radius=s.radius,
                    provenance=json.dumps(s.provenance),
                )
                for name in (
                    "vertices",
                    "faces",
                    "samples",
                    "areas",
                    "sdf",
                    "origin",
                    "inertia",
                ):
                    g.create_dataset(name, data=getattr(s, name), compression="gzip")

    @classmethod
    def load(cls, path):
        with h5py.File(path) as f:
            if f.attrs.get("format") != FORMAT or f.attrs.get("units") != "m":
                raise ValueError("Unsupported shape format or units")
            shapes = []
            for key in sorted(f, key=int):
                g = f[key]
                args = {
                    n: g[n][:]
                    for n in (
                        "vertices",
                        "faces",
                        "samples",
                        "areas",
                        "sdf",
                        "origin",
                        "inertia",
                    )
                }
                args.update(
                    {n: g.attrs[n] for n in ("id", "spacing", "volume", "radius")}
                )
                args["provenance"] = json.loads(g.attrs["provenance"])
                shapes.append(Shape(**args))
        return cls(shapes)


def from_mesh(
    mesh,
    *,
    shape_id="grain",
    unit,
    resolution=64,
    samples=2048,
    seed=0,
    provenance=None,
):
    if unit not in UNITS or resolution < 8 or samples < 16:
        raise ValueError(
            "Explicit m/mm/um unit, resolution >=8 and samples >=16 required"
        )
    mesh = mesh.copy()
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or len(mesh.split()) != 1
        or mesh.volume <= 0
    ):
        raise ValueError(
            "Mesh must be a single watertight, consistently wound, positive-volume solid"
        )
    if not np.isfinite(mesh.vertices).all() or mesh.area_faces.min() <= 0:
        raise ValueError("Mesh has nonfinite vertices or degenerate faces")
    mesh.apply_scale(UNITS[unit])
    centre = mesh.center_mass.copy()
    mesh.apply_translation(-centre)
    dx = float(mesh.extents.max() / (resolution - 1))
    origin = mesh.bounds[0] - 3 * dx
    dims = np.ceil(mesh.extents / dx).astype(int) + 7
    sdf = np.empty(tuple(dims), np.float32)
    # Proximity predicates have absolute tolerances. Query at order-one scale,
    # otherwise millimetre triangles can be misclassified as degenerate.
    proximity_scale = float(mesh.extents.max())
    proximity_mesh = mesh.copy()
    proximity_mesh.apply_scale(1 / proximity_scale)
    # Slabs bound proximity-query working memory; no source processing on the GPU.
    yz = np.stack(
        np.meshgrid(np.arange(dims[1]), np.arange(dims[2]), indexing="ij"), -1
    ).reshape(-1, 2)
    for x in range(dims[0]):
        points = origin + dx * np.column_stack([np.full(len(yz), x), yz])
        # trimesh uses positive inside, so negate explicitly.
        sdf[x] = (
            -trimesh.proximity.signed_distance(proximity_mesh, points / proximity_scale)
            * proximity_scale
        ).reshape(tuple(dims[1:]))
    # Stratified face-area sampling using a reproducible low-discrepancy sequence.
    from scipy.stats import qmc

    u = qmc.Sobol(3, scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(samples)))
    )[:samples]
    face = np.searchsorted(
        np.cumsum(mesh.area_faces) / mesh.area, u[:, 0], side="right"
    )
    tri = mesh.triangles[face]
    root = np.sqrt(u[:, 1])
    v = u[:, 2]
    pts = (
        (1 - root[:, None]) * tri[:, 0]
        + (root * (1 - v))[:, None] * tri[:, 1]
        + (root * v)[:, None] * tri[:, 2]
    )
    info = dict(
        provenance or {},
        input_unit=unit,
        source_to_body_translation_m=(-centre).tolist(),
        seed=seed,
        resolution=resolution,
        sample_count=samples,
        sdf_sign="negative_inside",
        sampling="scrambled_sobol_area",
        repairs=[],
    )
    s = Shape(
        shape_id,
        mesh.vertices.copy(),
        mesh.faces.copy(),
        pts,
        np.full(samples, mesh.area / samples),
        sdf,
        origin,
        dx,
        float(mesh.volume),
        mesh.moment_inertia.copy(),
        float(np.linalg.norm(mesh.vertices, axis=1).max()),
        info,
    )
    s.validate()
    return s


def prepare(path, output, *, unit, **kwargs):
    path = Path(path)
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("A single mesh file is required")
    # STL stores triangle soup: weld exactly equal coordinates, without rounding,
    # smoothing, reorienting faces or filling holes.
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices, inverse[mesh.faces], process=False)
    shape = from_mesh(
        mesh,
        unit=unit,
        shape_id=path.stem,
        provenance={
            "source_file": path.name,
            "source_sha256": sha256(path),
            "format_decode": "exact_coordinate_vertex_weld",
        },
        **kwargs,
    )
    ShapeLibrary([shape]).save(output)
    return shape


def sphere(radius=5e-4, resolution=32, samples=512):
    """Analytical verification fixture; its mesh is for display only."""
    if radius <= 0 or resolution < 8 or samples < 16:
        raise ValueError("Invalid sphere fixture")
    dx = 2 * radius / (resolution - 1)
    origin = np.full(3, -radius - 3 * dx)
    dims = resolution + 6
    grid = (
        np.stack(np.meshgrid(*([np.arange(dims)] * 3), indexing="ij"), -1) * dx + origin
    )
    sdf = (np.linalg.norm(grid, axis=-1) - radius).astype(np.float32)
    k = np.arange(samples)
    z = 1 - 2 * (k + 0.5) / samples
    angle = k * (np.pi * (3 - np.sqrt(5)))
    pts = radius * np.column_stack(
        [np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z]
    )
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    volume = 4 * np.pi * radius**3 / 3
    return Shape(
        "analytical-sphere",
        mesh.vertices,
        mesh.faces,
        pts,
        np.full(samples, 4 * np.pi * radius**2 / samples),
        sdf,
        origin,
        dx,
        volume,
        np.eye(3) * 0.4 * volume * radius**2,
        radius,
        {"fixture": "analytical_sphere", "radius_m": radius, "display_mesh_only": True},
    )


def with_face_quadrature(shape, rule="gauss3"):
    """Fixed source-triangle quadrature; preserves source geometry and physical data.

    Centroid integrates degree one; the symmetric three-point rule integrates
    degree two on each planar triangle. No claim about nonpolynomial or clipped
    contact integration follows from this polynomial property.
    """
    from dataclasses import replace

    if shape.provenance.get("display_mesh_only"):
        raise ValueError("Face quadrature requires physical source triangles")
    if rule == "centroid":
        barycentric = np.full((1, 3), 1 / 3)
        degree = 1
    elif rule == "gauss3":
        barycentric = np.full((3, 3), 1 / 6) + np.eye(3) / 2
        degree = 2
    else:
        raise ValueError("Unsupported triangle quadrature")
    triangles = np.asarray(shape.vertices)[shape.faces]
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    points = np.einsum("ij,fjk->fik", barycentric, triangles).reshape(-1, 3)
    weights = np.repeat(areas / len(barycentric), len(barycentric))
    result = replace(
        shape,
        samples=points,
        areas=weights,
        provenance=dict(
            shape.provenance,
            prior_sampling=shape.provenance.get("sampling"),
            sampling="source_triangle_" + rule,
            sample_count=len(points),
            quadrature_degree=degree,
            sample_key="original source face index * nodes_per_face + local node index",
        ),
    )
    result.validate()
    return result
