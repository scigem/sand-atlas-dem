"""Explicit artificial construction controls; no thermal material model is implied."""

import math

import numpy as np


def cooling_energy(step, warm_steps, cooling_steps, initial_energy):
    """Constant agitation followed by a quadratic decrease to zero."""
    if step < 0 or warm_steps < 0 or cooling_steps <= 0:
        raise ValueError("Invalid cooling schedule")
    if not math.isfinite(initial_energy) or initial_energy < 0:
        raise ValueError("Finite nonnegative agitation energy required")
    fraction = np.clip((step - warm_steps) / cooling_steps, 0, 1)
    return float(initial_energy * (1 - fraction) ** 2)


def agitation(mass, inertia, energy_per_body, rng):
    """Set equal total translational/rotational energy and zero linear momentum.

    Inertia is spatial, positive definite; all inputs use one consistent unit system.
    The scalar is total kinetic energy per body, not a temperature in kelvin.
    """
    mass = np.asarray(mass, float)
    inertia = np.asarray(inertia, float)
    n = len(mass)
    if n < 2 or inertia.shape != (n, 3, 3):
        raise ValueError("At least two masses and matching inertia tensors required")
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Positive finite masses required")
    if not np.isfinite(inertia).all() or not np.allclose(inertia, inertia.transpose(0, 2, 1)):
        raise ValueError("Finite symmetric inertia required")
    if np.any(np.linalg.eigvalsh(inertia) <= 0):
        raise ValueError("Positive definite inertia required")
    if not math.isfinite(energy_per_body) or energy_per_body < 0:
        raise ValueError("Finite nonnegative energy required")
    if energy_per_body == 0:
        return np.zeros((n, 3)), np.zeros((n, 3))
    velocity = rng.normal(size=(n, 3)) / np.sqrt(mass[:, None])
    velocity -= np.sum(mass[:, None] * velocity, axis=0) / mass.sum()
    velocity *= np.sqrt(n * energy_per_body / np.sum(mass[:, None] * velocity**2))
    momentum = np.einsum("nij,nj->ni", np.linalg.cholesky(inertia), rng.normal(size=(n, 3)))
    twice_energy = np.einsum("ni,ni->", momentum, np.linalg.solve(inertia, momentum[..., None])[..., 0])
    momentum *= np.sqrt(n * energy_per_body / twice_energy)
    return velocity, momentum


def select_timestep(elastic, viscous, cap, safety=0.5):
    """Select below the existing active-contact bound, retaining all device guards."""
    values = np.concatenate([np.ravel(elastic), np.ravel(viscous)])
    if np.isnan(values).any() or np.any(values <= 0):
        raise ValueError("Positive contact limits required (infinity means no contact)")
    if not math.isfinite(cap) or cap <= 0 or not 0 < safety < 1:
        raise ValueError("Invalid timestep cap or safety fraction")
    return min(cap, safety * float(values.min()))


def construction_quadrature(shape, count, seed):
    """Explicit area-stratified coarse quadrature on unchanged original mesh.

    This is only a packing aid. Original quadrature must check physical handoff.
    The selected original sample indices are retained, including any repeats.
    """
    from dataclasses import replace

    if type(count) is not int or count < 16:
        raise ValueError("At least16 construction samples required")
    if count >= len(shape.samples):
        return shape, np.arange(len(shape.samples))
    total = float(np.sum(shape.areas))
    cdf = np.cumsum(shape.areas, dtype=float) / total
    cdf[-1] = 1.
    quantiles = (np.arange(count) + np.random.default_rng(seed).random(count)) / count
    indices = np.searchsorted(cdf, quantiles)
    result = replace(shape, samples=shape.samples[indices].copy(),
                     areas=np.full(count, total/count), provenance={**shape.provenance,
                     "construction_quadrature": {"method": "area-stratified-original-node-subsampling",
                     "count": count, "seed": seed, "physical_handoff": "Original full quadrature mandatory"}})
    return result, indices
