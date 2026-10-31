"""Closed sphere/plane surface integrals for an explicit project wall variant.

Sphere centre is zero; inward wall normal is +z; offset=-R+depth.
Tangential spring displacement is uniform +x before pointwise Coulomb return.
These analytical oracles do not advance a DEM state.
"""

import numpy as np


def sphere_plane(radius, depth, kn, kt, mu, slip):
    if not np.isfinite([radius, depth, kn, kt, mu, slip]).all() or not (
        radius > 0 and 0 <= depth <= 2 * radius and kn > 0 and kt > 0 and mu >= 0
    ):
        raise ValueError("Finite sphere-cap domain and positive stiffness required")
    a, b = kt * abs(slip), mu * kn
    transition = min(depth, a / b) if b else depth
    j0 = b * transition**2 / 2 + a * (depth - transition)
    j1 = b * transition**3 / 3 + a * (depth**2 - transition**2) / 2
    prefactor = 2 * np.pi * radius
    sign = np.sign(slip)
    return dict(
        force=np.array([-sign * prefactor * j0, 0, np.pi * kn * radius * depth**2]),
        torque=np.array([0, sign * prefactor * ((radius - depth) * j0 + j1), 0]),
        normal_energy=np.pi * kn * radius * depth**3 / 3,
        contact_area=prefactor * depth,
    )
