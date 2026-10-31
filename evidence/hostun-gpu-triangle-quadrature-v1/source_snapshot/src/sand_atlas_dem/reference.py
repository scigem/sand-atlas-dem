"""Independent FP64 small-case reference. No Slang-generated equations are used."""

import numpy as np

from .util import finite_array, quat_matrix


def minimum_image(delta, cell):
    return delta - np.floor(delta / cell + 0.5) * cell


def transport_slip(slip, previous_normal, normal, spin=0.0):
    """Minimal normal rotation plus common-body spin; preserve elastic slip length."""
    a = np.asarray(previous_normal, float)
    b = np.asarray(normal, float)
    x = np.asarray(slip, float)
    if np.linalg.norm(a) < 0.5:
        return np.zeros(3)
    cross = np.cross(a, b)
    c = np.clip(a @ b, -1, 1)
    if c < -1 + 1e-7:
        # Contact normal flipped: prior local contact frame is no longer continuous.
        return np.zeros(3)
    out = x + np.cross(cross, x) + np.cross(cross, np.cross(cross, x)) / (1 + c)
    out = (
        out * np.cos(spin)
        + np.cross(b, out) * np.sin(spin)
        + b * (b @ out) * (1 - np.cos(spin))
    )
    return out - b * (b @ out)


def nodal_force(delta, normal, area, slip, relative_velocity, kn, kt, mu, dt):
    """Half-weighted directed traction law; source receives outward target normal."""
    if delta <= 0:
        return np.zeros(3), np.zeros(3)
    n = np.asarray(normal, float)
    vt = relative_velocity - n * np.dot(relative_velocity, n)
    trial = np.asarray(slip) + vt * dt
    elastic = -0.5 * kt * area * trial
    fn = 0.5 * kn * area * delta
    norm = np.linalg.norm(elastic)
    if norm > mu * fn and norm > 0:
        elastic *= mu * fn / norm
        trial = -elastic / (0.5 * kt * area)
    return fn * n + elastic, trial


def pair_contact(
    shape_i,
    shape_j,
    xi,
    xj,
    *,
    qi=(0, 0, 0, 1),
    qj=(0, 0, 0, 1),
    si=1.0,
    sj=1.0,
    kn=1.0,
    kt=1.0,
    mu=0.0,
    vi=None,
    vj=None,
    wi=None,
    wj=None,
    dt=0.0,
    history=None,
):
    """Symmetric two-pass reference; torques at a common spatial application point."""
    xi = finite_array(xi, (3,))
    xj = finite_array(xj, (3,))
    rotations = [quat_matrix(qi), quat_matrix(qj)]
    shapes = [shape_i, shape_j]
    positions = [xi, xj]
    scales = [si, sj]
    velocities = [
        np.zeros(3) if vi is None else np.asarray(vi),
        np.zeros(3) if vj is None else np.asarray(vj),
    ]
    spins = [
        np.zeros(3) if wi is None else np.asarray(wi),
        np.zeros(3) if wj is None else np.asarray(wj),
    ]
    force = np.zeros(3)
    ti = np.zeros(3)
    tj = np.zeros(3)
    state = {}
    work = 0.0
    count = 0
    for source in (0, 1):
        target = 1 - source
        pts = positions[source] + scales[source] * (
            shapes[source].samples @ rotations[source].T
        )
        local = (pts - positions[target]) @ rotations[target] / scales[target]
        phi, grad = shapes[target].query(local)
        for node in np.flatnonzero(phi < 0):
            g = rotations[target] @ grad[node]
            norm = np.linalg.norm(g)
            if norm < 1e-8:
                raise ValueError("Degenerate SDF gradient")
            n = g / norm
            p = pts[node]
            arms = [p - xi, p - xj]
            rv = (
                velocities[source]
                + np.cross(spins[source], arms[source])
                - velocities[target]
                - np.cross(spins[target], arms[target])
            )
            old = (history or {}).get((source, int(node)))
            slip = (
                np.zeros(3)
                if old is None
                else transport_slip(
                    old[0], old[1], n, 0.5 * np.dot(spins[0] + spins[1], n) * dt
                )
            )
            delta = -phi[node] * scales[target]
            area = shapes[source].areas[node] * scales[source] ** 2
            f, new = nodal_force(delta, n, area, slip, rv, kn, kt, mu, dt)
            sign = 1 if source == 0 else -1
            force += sign * f
            ti += np.cross(arms[0], sign * f)
            tj += np.cross(arms[1], -sign * f)
            state[(source, int(node))] = (new, n)
            work += 0.25 * kn * area * delta**2 + 0.25 * kt * area * np.dot(new, new)
            count += 1
    return dict(
        force=force,
        torque_i=ti,
        torque_j=tj,
        history=state,
        elastic_energy_proxy=work,
        contacts=count,
    )
