"""Cache correctness: exact arrays, input invalidation and damaged-entry recovery."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from sand_atlas_dem import mesh_cache
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.mesh import pack_meshes


@pytest.fixture
def library():
    return ShapeLibrary(
        [from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=65)]
    )


def test_hit_is_exact_and_skips_packing(library, tmp_path, monkeypatch):
    expected = pack_meshes(library, 0.001)
    cold, first = mesh_cache.packed_meshes(library, 0.001, directory=tmp_path)
    monkeypatch.setattr(
        mesh_cache, "pack_meshes", lambda *args: pytest.fail("Repacked on hit")
    )
    warm, second = mesh_cache.packed_meshes(library, 0.001, directory=tmp_path)
    assert first["status"] == "miss" and second["status"] == "hit"
    assert set(cold) == set(warm) == set(expected)
    for name in expected:
        assert expected[name].dtype == warm[name].dtype
        assert expected[name].shape == warm[name].shape
        assert expected[name].tobytes() == warm[name].tobytes() == cold[name].tobytes()


@pytest.mark.parametrize(
    "change", ["vertices", "faces", "samples", "units", "provenance", "order", "packer"]
)
def test_key_invalidates_inputs(library, change, monkeypatch):
    before = mesh_cache.identity(library, 0.001)[0]
    shape = library.shapes[0]
    unit = 0.001
    if change in {"vertices", "samples", "faces"}:
        array = getattr(shape, change).copy()
        if change == "faces":
            array = array[::-1].copy()
        else:
            array[0, 0] += 1e-9
        shape = replace(shape, **{change: array})
    elif change == "units":
        unit *= 2
    elif change == "provenance":
        shape = replace(shape, provenance={"source_sha256": "different-source"})
    elif change == "packer":
        monkeypatch.setattr(mesh_cache.np, "__version__", "different-packing-runtime")
    shapes = [shape] if change != "order" else [shape, replace(shape, id="second")]
    assert mesh_cache.identity(ShapeLibrary(shapes), unit)[0] != before


@pytest.mark.parametrize("damage", ["truncated", "array", "manifest", "missing"])
def test_damaged_entry_rebuilt_exactly(library, tmp_path, damage):
    expected, receipt = mesh_cache.packed_meshes(library, 0.001, directory=tmp_path)
    path = tmp_path / (receipt["key"] + ".npz")
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    else:
        with np.load(path, allow_pickle=False) as data:
            arrays = {n: data[n] for n in data.files}
        if damage == "array":
            arrays["meshClusterMin"][0, 0] += 1
        elif damage == "manifest":
            arrays["manifest"] = np.array("{}")
        else:
            del arrays["meshFrontier"]
        np.savez(path, **arrays)
    with pytest.warns(RuntimeWarning, match="Invalid mesh cache"):
        actual, receipt = mesh_cache.packed_meshes(library, 0.001, directory=tmp_path)
    assert receipt["status"] == "rebuilt-invalid"
    for name in expected:
        assert actual[name].tobytes() == expected[name].tobytes()


def test_display_mesh_never_accepted_from_cache(library, tmp_path):
    mesh_cache.packed_meshes(library, 0.001, directory=tmp_path)
    library.shapes[0].provenance["display_mesh_only"] = True
    with pytest.raises(ValueError, match="physical source"):
        mesh_cache.packed_meshes(library, 0.001, directory=tmp_path)


def test_explicit_disable(library, monkeypatch):
    monkeypatch.setenv("SAND_DEM_MESH_CACHE", "off")
    _, receipt = mesh_cache.packed_meshes(library, 0.001)
    assert receipt == {"status": "disabled"}
