"""Pinned source acquisition; no extraction, guessed units or geometry repair."""

import concurrent.futures
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
import uuid
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import trimesh

from .cases import register_material, resolve_material
from .geometry import ShapeLibrary, from_mesh
from .util import read_json, sha256, write_json


def source_recipe(name):
    path = Path(__file__).parent / "data" / "hostun-30.json"
    recipe = read_json(path)
    if name.strip().casefold() not in [
        recipe["id"],
        *[a.casefold() for a in recipe["aliases"]],
    ]:
        raise ValueError(
            f"No reviewed source/scale recipe for {name!r}; prepare geometry and register an explicit material receipt"
        )
    return recipe, path


def fetch_pinned(source, directory):
    """A failed or interrupted download never becomes a cached source."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source["filename"]
    if target.exists():
        if sha256(target) != source["sha256"]:
            raise ValueError(f"Cached source hash mismatch: {target}")
        return target
    # Unique partial files avoid clobbering a concurrent download's temporary data.
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".partial-", dir=directory)
    temporary = Path(temporary)
    try:
        request = Request(
            source["url"],
            headers={
                "User-Agent": "sand-atlas-dem/0.1 (+https://github.com/SciGEM/sand-atlas-dem)"
            },
        )
        with os.fdopen(fd, "wb") as stream, urlopen(request, timeout=60) as response:
            received = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                received += len(block)
                if received > source.get("bytes", 512 * 1024 * 1024):
                    raise ValueError("Downloaded source exceeds pinned byte budget")
                stream.write(block)
        if sha256(temporary) != source["sha256"]:
            raise ValueError(f"Downloaded source hash mismatch: {source['url']}")
        # Publish without overwriting a source installed by another process.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if sha256(target) != source["sha256"]:
                raise ValueError(f"Concurrent source hash mismatch: {target}")
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _save_library(library, target):
    temporary = target.with_suffix(".partial.h5")
    library.save(temporary)
    temporary.replace(target)


def _prepare_one(task):
    member, blob, descriptors, profile, directory, source = task
    name = member["name"]
    if hashlib.sha256(blob).hexdigest() != member["sha256"]:
        raise ValueError(f"Archive member hash mismatch: {name}")
    mesh = trimesh.load_mesh(io.BytesIO(blob), file_type="stl", process=False)
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices, inverse[mesh.faces], process=False)
    choices = sorted(
        (abs(mesh.volume / float(d["Volume (µm³)"]) - 1), d["Particle ID"])
        for d in descriptors
    )
    if choices[0][1] != member["descriptor_id"]:
        raise ValueError(f"Pinned descriptor/unit crosswalk changed: {name}")
    accepted = (
        len(mesh.split(only_watertight=False)) == 1
        and mesh.is_watertight
        and mesh.is_winding_consistent
        and mesh.volume > 0
    )
    if bool(accepted) != member["accepted"]:
        raise ValueError(f"Source topology decision changed: {name}")
    record = dict(
        member=member,
        relative_volume_difference=choices[0][0],
        next_best_volume_difference=choices[1][0],
        repairs=[],
    )
    if not accepted:
        return dict(
            record,
            status="excluded",
            reason="not_single_closed_consistent_positive_solid",
        )
    directory = Path(directory)
    target = directory / (Path(name).stem + ".h5")
    receipt = target.with_suffix(".json")
    provenance = dict(
        dataset=source["id"],
        source_url=source["archive"]["url"],
        source_archive_sha256=source["archive"]["sha256"],
        source_sha256=member["sha256"],
        archive_member=name,
        descriptor_table_sha256=source["descriptors"]["sha256"],
        descriptor_id=member["descriptor_id"],
        unit_basis=source["scale_evidence"],
        preparation_profile=profile,
        repairs=[],
    )
    if receipt.exists():
        cached = read_json(receipt)
        if cached["shape_library"] != str(target) or cached["member"] != member:
            raise ValueError(f"Prepared grain receipt mismatch: {target}")
        if sha256(target) != cached["shape_sha256"]:
            raise ValueError(f"Prepared grain hash mismatch: {target}")
        library = ShapeLibrary.load(target)
        if len(library.shapes) != 1 or library.shapes[0].provenance != provenance:
            raise ValueError(f"Prepared grain provenance mismatch: {target}")
        return cached
    if target.exists():
        raise ValueError(
            f"Unreceipted grain asset exists: {target}; retain it and use a fresh cache"
        )
    shape = from_mesh(
        mesh,
        shape_id="hostun-" + Path(name).stem,
        unit="um",
        resolution=profile["resolution"],
        samples=profile["samples"],
        seed=profile["seed"],
        provenance=provenance,
    )
    _save_library(ShapeLibrary([shape]), target)
    record.update(
        status="prepared", shape_library=str(target), shape_sha256=sha256(target)
    )
    write_json(receipt, record)
    return record


def prepare_material(
    name, cache, *, resolution=64, samples=2048, workers=4, source_directory=None
):
    """Prepare or resume a reviewed material; register only a complete library."""
    for key, value, minimum in [
        ("resolution", resolution, 8),
        ("samples", samples, 16),
        ("workers", workers, 1),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    recipe, recipe_path = source_recipe(name)
    cache = Path(cache).resolve()
    profile = dict(
        format="sand-dem-preparation-profile/1",
        resolution=resolution,
        samples=samples,
        seed=0,
        recipe_sha256=sha256(recipe_path),
        code_sha256={
            p.name: sha256(p)
            for p in [Path(__file__), Path(__file__).with_name("geometry.py")]
        },
        versions={
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "trimesh", "h5py", "rtree"]
        },
    )
    identity = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    registered = cache / (recipe["id"] + ".json")
    if registered.exists():
        record = resolve_material(recipe["id"], cache)
        if record["source"].get("preparation_identity") != identity:
            raise ValueError(
                "Material is registered with another preparation profile; use a different --cache directory"
            )
        report_path = Path(record["source"]["preparation_report"])
        if sha256(report_path) != record["source"]["preparation_report_sha256"]:
            raise ValueError("Preparation report hash mismatch")
        return record
    private = cache / "private" / recipe["id"]
    sources = (
        Path(source_directory).resolve() if source_directory else private / "sources"
    )
    archive = fetch_pinned(recipe["archive"], sources)
    table = fetch_pinned(recipe["descriptors"], sources)
    with table.open() as stream:
        descriptors = list(csv.DictReader(stream))
    ids = [m["descriptor_id"] for m in recipe["members"]]
    if (
        len(set(ids)) != len(ids)
        or set(ids) != {d["Particle ID"] for d in descriptors}
        or len(ids) != len(descriptors)
    ):
        raise ValueError("Nonunique or incomplete pinned descriptor crosswalk")
    directory = private / identity
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive writer; a crashed process leaves a visible lock for explicit recovery.
    lock = directory / "preparation.lock"
    with lock.open("x") as stream:
        json.dump(dict(pid=os.getpid(), identity=identity), stream)
    report_path = directory / ("attempt-" + uuid.uuid4().hex + ".json")
    report = dict(
        format="sand-dem-material-preparation/1",
        status="running",
        profile=profile,
        recipe=recipe,
        grains=[],
        scientific_admission=False,
    )
    try:
        write_json(report_path, report)
        with zipfile.ZipFile(archive) as z:
            names = [i.filename for i in z.infolist() if not i.is_dir()]
            if len(set(names)) != len(names) or set(names) != {
                m["name"] for m in recipe["members"]
            }:
                raise ValueError(
                    "Archive member inventory differs from the reviewed recipe"
                )
            tasks = [
                (m, z.read(m["name"]), descriptors, profile, str(directory), recipe)
                for m in recipe["members"]
            ]
        # CPU-only geometry processing; never executes downloaded code.
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for record in executor.map(_prepare_one, tasks):
                report["grains"].append(record)
                write_json(report_path, report)
        prepared = [r for r in report["grains"] if r["status"] == "prepared"]
        library = directory / "shapes.h5"
        if library.exists():
            old_receipt = read_json(directory / "material.json")
            if sha256(library) != old_receipt["shape_sha256"]:
                raise ValueError("Existing combined library hash mismatch")
            report.update(
                status="reused_complete",
                original_report=old_receipt["source"]["preparation_report"],
            )
            write_json(report_path, report)
            return register_material(directory / "material.json", cache)
        _save_library(
            ShapeLibrary(
                [ShapeLibrary.load(r["shape_library"]).shapes[0] for r in prepared]
            ),
            library,
        )
        report.update(
            status="complete",
            library_sha256=sha256(library),
            accepted=len(prepared),
            excluded=len(tasks) - len(prepared),
        )
        write_json(report_path, report)
        receipt = directory / "material.json"
        write_json(
            receipt,
            dict(
                format="sand-dem-material/1",
                id=recipe["id"],
                aliases=recipe["aliases"],
                shape_library=str(library),
                shape_sha256=sha256(library),
                scale_evidence=recipe["scale_evidence"],
                source=dict(
                    archive=recipe["archive"],
                    descriptors=recipe["descriptors"],
                    preparation_identity=identity,
                    preparation_profile=profile,
                    preparation_report=str(report_path),
                    preparation_report_sha256=sha256(report_path),
                ),
            ),
        )
        return register_material(receipt, cache)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(report_path, report)
        raise
    finally:
        lock.unlink()
