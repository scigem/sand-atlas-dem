"""Independent continuum sphere integral versus the sampled SDF contact law.

This is a verification study, not Hertz theory and not experimental validation.
The analytic integrand is derived independently from spherical surface geometry.
"""

import json

import numpy as np
from scipy.integrate import quad

from sand_atlas_dem.geometry import sphere
from sand_atlas_dem.reference import pair_contact
from sand_atlas_dem.util import write_json


def study(output):
    radius = 1.0
    separation = 1.8
    kn = 1.0

    def integrand(c):
        q = np.sqrt(radius**2 + separation**2 - 2 * radius * separation * c)
        return (radius - q) * (radius * c - separation) / q

    exact = (
        2
        * np.pi
        * radius**2
        * kn
        * quad(integrand, separation / (2 * radius), 1, epsabs=1e-13)[0]
    )
    rows = []
    for resolution in (32, 64, 128):
        for samples in (2048, 8192, 32768):
            shape = sphere(radius=radius, resolution=resolution, samples=samples)
            result = pair_contact(
                shape, shape, np.zeros(3), np.array([separation, 0, 0]), kn=kn
            )
            error = abs(result["force"][0] - exact) / abs(exact)
            rows.append(
                dict(
                    resolution=resolution,
                    samples=samples,
                    force=result["force"].tolist(),
                    relative_error=error,
                )
            )
    finest = rows[-1]
    report = dict(
        format="sand-dem-contact-refinement/1",
        reference="independent spherical surface integral of half-weighted two-pass traction law",
        geometry=dict(radius=radius, separation=separation, normal_stiffness=kn),
        reference_force_x=exact,
        rows=rows,
        policy=dict(finest_reference_relative_error=0.02),
        passed=bool(finest["relative_error"] <= 0.02),
    )
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(study("evidence/contact-refinement-v1.json"))
