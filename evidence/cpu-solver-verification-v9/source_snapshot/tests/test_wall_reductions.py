"""Multiwall/multiparticle tests exercise reductions and shared history keys."""

import os

import numpy as np
import pytest

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.benchmark import memory_report
from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.gpu import Engine, f4
from sand_atlas_dem.spec import Experiment

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
    ),
]


def plane(normal, offset, velocity=(0, 0, 0)):
    return dict(
        normal=normal,
        offset=offset,
        velocity=list(velocity),
        normal_stiffness=1e12,
        tangential_stiffness=5e11,
        friction=0.3,
    )


def test_two_stage_wall_reduction_preserves_exact_integer_counts_and_moments():
    shape = sphere(resolution=16, samples=2048)
    n = 257  # Two partial groups, followed by a second reduction pass.
    centers = np.array(
        [[0.002 + (i % 17) * 0.002, 0.002 + (i // 17) * 0.002, 0.002] for i in range(n)]
    )
    walls = [plane([0, 0, 1], 0.00155), plane([0, 0, -1], -0.00245)]
    spec = Experiment(
        periodic=False,
        particles=n,
        timestep=1e-8,
        max_overlap_fraction=0.1,
        contacts_per_particle=256,
        planes=walls,
    )
    engine = Engine(
        ShapeLibrary([shape]), spec, centers, [0.04, 0.04, 0.006], [0] * n, [1] * n
    )
    engine.initialize()
    row = engine.diagnostics()
    assert len(engine.wall_diag_buffers) == 2
    for index, wall in enumerate(walls):
        normal = np.array(wall["normal"])
        depth = np.maximum(0, wall["offset"] - (shape.samples + centers[0]) @ normal)
        forces = shape.areas[:, None] * 1e12 * depth[:, None] * normal
        reaction = -n * forces.sum(0)
        # Sum over every original node and every body centre independently.
        moment = -np.cross(shape.samples, forces).sum(0) * n - np.cross(
            centers, forces.sum(0)
        ).sum(0)
        value = row["walls"][index]
        assert value["active_surface_contacts"] == n * np.count_nonzero(depth)
        assert (
            np.linalg.norm(value["reaction"] - reaction)
            < np.linalg.norm(reaction) * 1e-4
        )
        assert (
            np.linalg.norm(value["reaction_moment_origin"] - moment)
            < np.linalg.norm(reaction) * 0.04 * 1e-4
        )
    assert (
        sum(w["active_surface_contacts"] for w in row["walls"])
        == row["active_surface_contacts"]
    )
    inventory = memory_report(engine)
    # Compare with direct traversal of every owned Slang buffer, including the
    # reduction hierarchy; this catches an omitted wall allocation category.
    seen = {}

    def collect(value):
        if isinstance(value, engine.m.spy.Buffer):
            seen[id(value)] = value.size
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif type(value).__name__ == "Scan":
            collect(vars(value))

    collect(vars(engine))
    assert sum(seen.values()) == inventory["logical_solver_buffer_bytes"]


def test_wall_and_grain_histories_coexist_and_wall_work_reduces_over_bodies():
    shape = sphere(resolution=32, samples=8192)
    walls = [plane([0, 0, 1], 0.00155)]
    spec = Experiment(
        periodic=False,
        particles=2,
        timestep=1e-8,
        max_overlap_fraction=0.1,
        contacts_per_particle=4096,
        planes=walls,
    )
    positions = [[0.002] * 3, [0.002995, 0.002, 0.002]]
    engine = Engine(
        ShapeLibrary([shape]),
        spec,
        positions,
        [0.006] * 3,
        [0, 0],
        [1, 1],
        metal=Metal(),
    )
    engine.initialize()
    engine.buffers["velocity"].copy_from_numpy(
        f4(
            np.array([[0, 0.01, 0], [0, -0.01, 0]])
            * engine.time_unit
            / engine.length_unit
        )
    )
    engine.advance(3, wall_velocities=[[0.1, 0, 0.1]])
    row, state = engine.diagnostics(), engine.state()
    assert row["active_grain_pairs"] == 1
    wall_keys = state["meta"][:, 1] == 2
    assert wall_keys.any() and (~wall_keys).any()
    assert len({tuple(k[:3]) for k in state["meta"]}) == state["contacts"]
    assert np.max(np.abs(state["slip"][wall_keys, :3])) > 0
    assert np.max(np.abs(state["slip"][~wall_keys, :3])) > 0
    scale = row["elastic_energy_proxy"]
    assert abs(row["wall_work"] - row["walls"][0]["work"]) < scale * 1e-5
    total_force = state["force"][:, :3].sum(0) * engine.force_unit
    assert (
        np.linalg.norm(total_force + row["walls"][0]["reaction"])
        < np.linalg.norm(total_force) * 1e-5
    )
    total_moment = np.cross(
        (state["position"][:, :3] + state["positionRemainder"][:, :3])
        * engine.length_unit,
        state["force"][:, :3] * engine.force_unit,
    ).sum(0)
    total_moment += state["torque"][:, :3].sum(0) * engine.energy_unit
    assert (
        np.linalg.norm(total_moment + row["walls"][0]["reaction_moment_origin"])
        < np.linalg.norm(total_force) * 0.006 * 1e-5
    )
