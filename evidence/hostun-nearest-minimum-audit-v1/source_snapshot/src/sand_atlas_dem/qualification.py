"""Frozen, independent small-case engineering qualification on the actual GPU."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from .backend import Metal
from .geometry import ShapeLibrary, from_mesh, sphere
from .gpu import Engine, f4
from .reference import pair_contact
from .spec import Experiment
from .util import read_json, sha256, write_json


def advance_chunked(engine, steps, **kwargs):
    for start in range(0, steps, 100):
        engine.advance(min(100, steps - start), **kwargs)


def evaluate(engine, dt):
    """Prescribed-pose contact update, for kinematic tests only; no dynamics step."""
    encoder = engine.m.device.create_command_encoder()
    engine.forces(encoder, engine.bindings(dt=dt))
    engine.m.submit(encoder)
    engine.m.device.wait()
    engine.check_errors()
    engine.active = 1 - engine.active
    engine.history = True


def pair(metal, shape, *, dt=1e-8, contacts=256, separation=0.99, friction=10):
    d = (6 * shape.volume / np.pi) ** (1 / 3)
    positions = np.array([[3 * d, 3 * d, 3 * d], [(3 + separation) * d, 3 * d, 3 * d]])
    spec = replace(
        Experiment(),
        particles=2,
        timestep=dt,
        contacts_per_particle=contacts,
        max_overlap_fraction=0.2,
        friction=friction,
    )
    e = Engine(
        ShapeLibrary([shape]),
        spec,
        positions,
        [10 * d] * 3,
        [0, 0],
        [1, 1],
        metal=metal,
    )
    e.initialize()
    return e


def keyed_history(state):
    return {
        tuple(k[:3]): np.array(v[:3], float)
        for k, v in zip(state["meta"], state["slip"])
    }


def force(engine):
    return engine.m.read(engine.buffers["force"]).reshape(-1, 4)[:, :3].astype(float)


def objectivity(metal, shape, policy):
    rows = []
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    for increments in policy["increments"]:
        e = pair(metal, shape)
        v = np.zeros((2, 4), np.float32)
        v[0, 1] = 0.1
        e.buffers["velocity"].copy_from_numpy(v)
        evaluate(e, e.spec.timestep)
        start = e.state()
        initial = keyed_history(start)
        f0 = force(e)
        center = start["position"][:, :3].mean(0).astype(float)
        oldx = start["position"][:, :3].astype(float) - center
        angle = policy["angle"] / increments
        angular = axis * angle / (e.spec.timestep / e.time_unit)
        # A sphere's inertia is isotropic, so common rigid angular velocity is exact.
        inertia = (
            np.eye(3)
            * 0.4
            * e.body[0, 1]
            * (shape.provenance["radius_m"] / e.length_unit) ** 2
        )
        for step in range(1, increments + 1):
            rot = Rotation.from_rotvec(axis * angle * step)
            x = oldx @ rot.as_matrix().T
            e.buffers["position"].copy_from_numpy(f4(x + center))
            e.buffers["rotation"].copy_from_numpy(
                np.tile(rot.as_quat(), (2, 1)).astype(np.float32)
            )
            e.buffers["velocity"].copy_from_numpy(f4(np.cross(angular, x)))
            e.buffers["momentum"].copy_from_numpy(
                f4(np.tile(inertia @ angular, (2, 1)))
            )
            evaluate(e, e.spec.timestep)
        end = e.state()
        actual = keyed_history(end)
        rot = Rotation.from_rotvec(axis * policy["angle"]).as_matrix()
        force_error = np.linalg.norm(force(e) - f0 @ rot.T) / np.linalg.norm(f0)
        common = set(initial) & set(actual)
        expected = np.array([rot @ initial[k] for k in sorted(common)])
        got = np.array([actual[k] for k in sorted(common)])
        error = np.linalg.norm(got - expected) / np.linalg.norm(expected)
        rows.append(
            dict(
                increments=increments,
                force_relative_error=float(force_error),
                history_relative_error=float(error),
                same_contact_keys=set(initial) == set(actual),
            )
        )
    last = rows[-1]
    return dict(
        rows=rows,
        passed=last["force_relative_error"] <= policy["relative_force_limit"]
        and last["history_relative_error"] <= policy["relative_history_limit"]
        and all(r["same_contact_keys"] for r in rows),
    )


def separation(metal, shape, policy):
    e = pair(metal, shape)
    v = np.zeros((2, 4), np.float32)
    v[0, 1] = 0.1
    e.buffers["velocity"].copy_from_numpy(v)
    evaluate(e, e.spec.timestep)
    original = e.state()
    seed = float(np.linalg.norm(original["slip"][:, :3]))
    x = original["position"].copy()
    x[1, 0] = x[0, 0] + 1.02
    e.buffers["position"].copy_from_numpy(x)
    evaluate(e, 0)
    absent = e.state()["contacts"]
    e.buffers["velocity"].copy_from_numpy(np.zeros((2, 4), np.float32))
    e.buffers["position"].copy_from_numpy(original["position"])
    evaluate(e, 0)
    returned = e.state()
    slip = float(np.linalg.norm(returned["slip"][:, :3]))
    return dict(
        seeded_slip_norm=seed,
        separated_contacts=absent,
        returned_contacts=returned["contacts"],
        returned_slip_norm=slip,
        passed=seed > 0
        and absent == 0
        and returned["contacts"] > 0
        and slip <= policy["returned_slip_limit"],
    )


def rolling(metal, shape, policy):
    """Instantaneous nominal pure rolling; finite contact patches can have creep."""
    e = pair(metal, shape, friction=0.5)
    physical_w = np.array(policy["angular_velocity_rad_s"])
    physical_v = -np.cross(physical_w, [shape.provenance["radius_m"], 0, 0])
    v = np.zeros((2, 3))
    v[0] = physical_v * e.time_unit / e.length_unit
    inertia = shape.inertia * e.spec.density / (e.mass_unit * e.length_unit**2)
    momentum = np.zeros((2, 3))
    momentum[0] = inertia @ (physical_w * e.time_unit)
    e.buffers["velocity"].copy_from_numpy(f4(v))
    e.buffers["momentum"].copy_from_numpy(f4(momentum))
    positions = e.state()["position"][:, :3] * e.length_unit
    reference = pair_contact(
        shape,
        shape,
        *positions,
        vi=physical_v,
        wi=physical_w,
        kn=e.spec.normal_stiffness,
        kt=e.spec.tangential_stiffness,
        mu=e.spec.friction,
        dt=e.spec.timestep,
    )
    evaluate(e, e.spec.timestep)
    state = e.state()
    actual = force(e)[0] * e.force_unit
    torque = e.m.read(e.buffers["torque"]).reshape(-1, 4)[:, :3] * e.energy_unit
    fe = np.linalg.norm(actual - reference["force"]) / np.linalg.norm(
        reference["force"]
    )
    te = np.linalg.norm(
        torque - np.array([reference["torque_i"], reference["torque_j"]])
    ) / (np.linalg.norm(reference["force"]) * e.length_unit)
    expected = np.array(
        [reference["history"][(int(k[0]), int(k[2]))][0] for k in state["meta"]]
    )
    slip = state["slip"][:, :3] * e.length_unit
    he = np.linalg.norm(slip - expected) / max(np.linalg.norm(expected), 1e-30)
    return dict(
        relative_force_error=float(fe),
        relative_torque_error=float(te),
        relative_history_error=float(he),
        contacts=state["contacts"],
        passed=fe <= policy["relative_force_limit"]
        and te <= policy["relative_torque_limit"]
        and he <= policy["relative_history_limit"],
    )


def growth(metal, shape, policy):
    engines = []
    for cap in [policy["initial_contacts_per_particle"], 256]:
        e = pair(metal, shape, contacts=cap, separation=1.001, friction=0)
        v = np.zeros((2, 4), np.float32)
        v[0, 0] = 0.0305 / (e.spec.timestep / e.time_unit)
        v[1, 0] = -v[0, 0]
        e.buffers["velocity"].copy_from_numpy(v)
        e.advance(1)
        engines.append(e)
    small, large = engines
    a = small.state()
    b = large.state()
    errors = {
        name: float(
            np.linalg.norm(a[name] - b[name]) / max(np.linalg.norm(b[name]), 1e-20)
        )
        for name in ("position", "velocity", "force", "torque")
    }
    return dict(
        retries=small.capacity_retries,
        contacts=a["contacts"],
        relative_state_errors=errors,
        passed=bool(small.capacity_retries)
        and a["contacts"] == b["contacts"]
        and max(errors.values()) <= policy["relative_state_limit"],
    )


def rotation(metal, policy):
    shape = from_mesh(
        trimesh.creation.box(extents=policy["extents_m"]),
        unit="m",
        resolution=16,
        samples=64,
    )
    rows = []
    for dt in policy["timesteps_s"]:
        e = pair(metal, shape, dt=dt, separation=5)
        inertia = shape.inertia * e.spec.density / (e.mass_unit * e.length_unit**2)
        angular = np.array(policy["angular_velocity_rad_s"]) * e.time_unit
        l = (inertia @ angular).astype(np.float32).astype(float)
        e.buffers["momentum"].copy_from_numpy(f4(np.tile(l, (2, 1))))
        inv = np.linalg.inv(inertia)

        def derivative(t, q):
            matrix = Rotation.from_quat(q).as_matrix()
            w = matrix @ inv @ matrix.T @ l
            return 0.5 * np.r_[q[3] * w + np.cross(w, q[:3]), -np.dot(w, q[:3])]

        steps = round(policy["duration_s"] / dt)
        duration = steps * dt / e.time_unit
        ref = solve_ivp(
            derivative,
            [0, duration],
            [0, 0, 0, 1],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        ).y[:, -1]
        before = e.diagnostics()
        advance_chunked(e, steps)
        after = e.diagnostics()
        state = e.state()
        error = float(
            (
                Rotation.from_quat(state["rotation"][0]) * Rotation.from_quat(ref).inv()
            ).magnitude()
        )
        energy = abs(after["kinetic_energy"] / before["kinetic_energy"] - 1)
        rows.append(
            dict(
                timestep_s=dt,
                steps=steps,
                orientation_error_rad=error,
                relative_energy_drift=float(energy),
                spatial_momentum_unchanged=bool(
                    np.array_equal(
                        state["momentum"][:, :3], np.tile(l, (2, 1)).astype(np.float32)
                    )
                ),
            )
        )
    refinement_ratio = rows[0]["orientation_error_rad"] / max(
        rows[-1]["orientation_error_rad"], 1e-30
    )
    return dict(
        rows=rows,
        refinement_ratio=refinement_ratio,
        passed=refinement_ratio >= policy.get("minimum_refinement_ratio", 0)
        and all(r["spatial_momentum_unchanged"] for r in rows)
        and rows[-1]["orientation_error_rad"] <= policy["orientation_error_rad_limit"]
        and max(r["relative_energy_drift"] for r in rows)
        <= policy["relative_energy_drift_limit"],
    )


def affine_work(metal, shape, policy):
    e = pair(metal, shape, separation=3, dt=policy["timestep_s"])
    v = np.array([[0.1, 0.2, 0.3], [-0.2, 0.3, -0.1]])
    e.buffers["velocity"].copy_from_numpy(f4(v))
    before = e.diagnostics()
    advance_chunked(e, policy["steps"], rate=policy["rate_s"])
    after = e.diagnostics()
    expected = v * np.exp(
        -np.array(policy["rate_s"]) * policy["steps"] * policy["timestep_s"]
    )
    actual = e.state()["velocity"][:, :3]
    ve = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    delta = after["kinetic_energy"] - before["kinetic_energy"]
    work = after["cell_work"]
    we = float(abs(work - delta) / abs(delta))
    return dict(
        relative_velocity_error=ve,
        relative_work_error=we,
        kinetic_change_J=delta,
        cell_work_J=work,
        passed=ve <= policy["relative_velocity_limit"]
        and we <= policy["relative_work_limit"]
        and after["active_surface_contacts"] == 0,
    )


def static_cell_energy(metal, shape, policy):
    rows = []
    for dt in policy["timesteps_s"]:
        e = pair(metal, shape, dt=dt, friction=0)
        before = e.diagnostics()
        initial = before["kinetic_energy"] + before["elastic_energy_proxy"]
        steps = round(policy["duration_s"] / dt)
        errors = []
        for start in range(0, steps, 100):
            e.advance(min(100, steps - start))
            r = e.diagnostics()
            errors.append(
                abs((r["kinetic_energy"] + r["elastic_energy_proxy"]) / initial - 1)
            )
        rows.append(
            dict(
                timestep_s=dt,
                steps=steps,
                maximum_relative_proxy_energy_drift=float(max(errors)),
            )
        )
    return dict(
        rows=rows,
        passed=max(r["maximum_relative_proxy_energy_drift"] for r in rows)
        <= policy["relative_proxy_drift_limit"],
    )


def qualify(protocol_path, output):
    policy = read_json(protocol_path)
    if policy.get("format") != "sand-dem-mechanics-protocol/1":
        raise ValueError("Unknown protocol")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        format="sand-dem-mechanics-study/1",
        protocol=policy,
        protocol_sha256=sha256(protocol_path),
        source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        shader_hashes={
            p.name: sha256(p)
            for p in (Path(__file__).parent / "shaders").glob("*.slang")
        },
        cases={},
        status="running",
    )
    write_json(output / "report.json", report)
    m = Metal()
    report["backend"] = m.doctor()
    shape = sphere(**policy["sphere"])
    for name, fn in [
        ("objectivity", lambda: objectivity(m, shape, policy["objectivity"])),
        ("separation", lambda: separation(m, shape, policy["separation"])),
        ("rolling", lambda: rolling(m, shape, policy["rolling"])),
        ("growth", lambda: growth(m, shape, policy["growth"])),
        ("rotation", lambda: rotation(m, policy["rotation"])),
        ("affine_work", lambda: affine_work(m, shape, policy["affine_work"])),
        (
            "static_cell_energy",
            lambda: static_cell_energy(m, shape, policy["static_cell_energy"]),
        ),
    ]:
        if name == "rolling" and name not in policy:
            continue
        start = time.perf_counter()
        try:
            result = fn()
            if not isinstance(result.get("passed"), (bool, np.bool_)):
                raise TypeError("Study gate must return a boolean")
            result["passed"] = bool(result["passed"])
        except Exception as exc:
            result = dict(passed=False, failure=f"{type(exc).__name__}: {exc}")
        result["wall_seconds"] = time.perf_counter() - start
        report["cases"][name] = result
        write_json(output / "report.json", report)
        print(name, result.get("passed"), flush=True)
    report["status"] = (
        "passed" if all(r["passed"] for r in report["cases"].values()) else "failed"
    )
    write_json(output / "report.json", report)
    return report
