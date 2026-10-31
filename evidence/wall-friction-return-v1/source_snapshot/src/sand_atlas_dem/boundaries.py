"""Strict SI input contract for fixed-normal translating penalty planes."""

import numpy as np

from .util import finite_array

PLANE_FIELDS = {
    "normal",
    "offset",
    "velocity",
    "normal_stiffness",
    "tangential_stiffness",
    "friction",
}
BOX_NORMALS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
)


def validate_planes(planes, periodic, boundary_box):
    if not isinstance(planes, list) or (planes and periodic):
        raise ValueError("Planes require a nonperiodic domain and a list")
    for plane in planes:
        if not isinstance(plane, dict) or set(plane) != PLANE_FIELDS:
            raise ValueError(
                "Every plane requires explicit normal, offset, velocity and contact properties"
            )
        normal = finite_array(plane["normal"], (3,))
        if abs(np.linalg.norm(normal) - 1) > 1e-10:
            raise ValueError(
                "Plane normal must have unit length; automatic normalization is forbidden"
            )
        finite_array(plane["velocity"], (3,))
        finite_array(plane["offset"], ())
        for name in ("normal_stiffness", "tangential_stiffness", "friction"):
            value = finite_array(plane[name], ())
            if value < 0 or (name != "friction" and value == 0):
                raise ValueError(
                    "Plane stiffness must be positive and friction nonnegative"
                )
    if boundary_box is not None:
        if (
            not isinstance(boundary_box, list)
            or len(boundary_box) != 6
            or any(type(i) is not int or not 0 <= i < len(planes) for i in boundary_box)
            or len(set(boundary_box)) != 6
        ):
            raise ValueError("boundary_box requires six distinct plane indices")
        normals = [planes[i]["normal"] for i in boundary_box]
        if not np.array_equal(normals, BOX_NORMALS):
            raise ValueError(
                "boundary_box planes must be ordered xmin,xmax,ymin,ymax,zmin,zmax with exact inward axis normals"
            )
        box_extents(np.array([p["offset"] for p in planes]), boundary_box)


def box_extents(offsets, indices, domain=None):
    """Measurement geometry only; particle/contact dynamics remain on the GPU."""
    values = finite_array(offsets)[indices]
    low, high = values[::2], -values[1::2]
    if np.any(high <= low):
        raise ValueError("Boundary box has a nonpositive gap")
    if domain is not None and (np.any(low < 0) or np.any(high > domain)):
        raise ValueError("Boundary box must lie inside the computational domain")
    return high - low
