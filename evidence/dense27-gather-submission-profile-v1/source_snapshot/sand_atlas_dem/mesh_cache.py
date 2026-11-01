"""Verified, private, content-addressed cache of immutable mesh packing arrays."""

import hashlib
import json
import os
import platform
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import scipy

from .mesh import pack_meshes

FORMAT = "sand-dem-mesh-cache/1"
UINT = {
    "meshFrontier",
    "meshClusterInfo",
    "meshClusterRanges",
    "meshSampleIndices",
    "meshHullInfo",
    "meshRoots",
    "meshLinks",
}
FLOAT = {
    "meshClusterMin",
    "meshClusterMax",
    "meshPlanes",
    "meshMin",
    "meshMax",
    "meshTriangles",
    "meshNormals",
}


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, a.shape]).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def identity(library, length_unit):
    """Hash actual packer inputs; never trust a source filename or claimed hash."""
    if not np.isfinite(length_unit) or length_unit <= 0:
        raise ValueError("Mesh length unit must be finite and positive")
    shapes = []
    for shape in library.shapes:
        if shape.provenance.get("display_mesh_only"):
            raise ValueError("Mesh query requires physical source triangles")
        shapes.append(
            dict(
                id=shape.id,
                provenance=shape.provenance,
                arrays={
                    name: array_hash(getattr(shape, name))
                    for name in ("vertices", "faces", "samples")
                },
            )
        )
    record = dict(
        format=FORMAT,
        length_unit=float(length_unit).hex(),
        shapes=shapes,
        packer_sha256=hashlib.sha256(
            Path(__file__).with_name("mesh.py").read_bytes()
        ).hexdigest(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        machine=platform.machine(),
    )
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    key = hashlib.sha256(encoded.encode()).hexdigest()
    # JSON-equivalent provenance (for example tuple/list) must compare on a hit.
    return key, json.loads(encoded)


def cache_directory():
    value = os.environ.get("SAND_DEM_MESH_CACHE")
    if value == "off":
        return None
    if value:
        return Path(value).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "sand-atlas-dem" / "mesh"


def _load(path, record):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest["identity"] != record or set(archive.files) != UINT | FLOAT | {
            "manifest"
        }:
            raise ValueError("Mesh cache identity or array set mismatch")
        arrays = {name: archive[name] for name in UINT | FLOAT}
    for name, a in arrays.items():
        dtype = np.uint32 if name in UINT else np.float32
        shape_ok = (
            a.ndim == 1
            if name == "meshSampleIndices"
            else a.ndim == 2 and a.shape[1] == 4
        )
        if (
            a.dtype != dtype
            or not shape_ok
            or array_hash(a) != manifest["arrays"][name]
        ):
            raise ValueError(f"Mesh cache array verification failed: {name}")
    return arrays


def packed_meshes(library, length_unit, *, directory=None):
    """Return packed arrays and a receipt; invalid/unwritable caches rebuild visibly."""
    directory = cache_directory() if directory is None else Path(directory)
    if directory is None:
        return pack_meshes(library, length_unit), dict(status="disabled")
    key, record = identity(library, length_unit)
    path = directory / (key + ".npz")
    status = "miss"
    try:
        if path.exists():
            return _load(path, record), dict(status="hit", key=key, path=str(path))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        zipfile.BadZipFile,
    ) as error:
        warnings.warn(f"Invalid mesh cache; rebuilding {path}: {error}", RuntimeWarning)
        status = "rebuilt-invalid"
    arrays = pack_meshes(library, length_unit)
    manifest = json.dumps(
        dict(identity=record, arrays={n: array_hash(a) for n, a in arrays.items()}),
        sort_keys=True,
    )
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Same-filesystem atomic replacement; readers see a complete entry only.
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, manifest=np.array(manifest), **arrays)
        os.replace(temporary, path)
    except OSError as error:
        warnings.warn(f"Mesh cache could not be saved: {error}", RuntimeWarning)
        status = "uncached"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return arrays, dict(status=status, key=key, path=str(path))
