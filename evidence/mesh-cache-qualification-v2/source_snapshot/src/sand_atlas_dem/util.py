"""Stable records and explicit conventions."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
            default=_json_value,
        )
        + "\n"
    )


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON: {x}")
        ),
    )


def finite_array(value, shape=None, positive=False):
    a = np.asarray(value, dtype=np.float64)
    if (
        (shape is not None and a.shape != shape)
        or not np.isfinite(a).all()
        or (positive and np.any(a <= 0))
    ):
        raise ValueError(
            f"Expected finite {'positive ' if positive else ''}array with shape {shape}"
        )
    return a


def quat_matrix(q):
    """Body to spatial rotation; quaternion order x,y,z,w."""
    x, y, z, w = finite_array(q, (4,))
    if abs(np.dot(q, q) - 1) > 1e-5:
        raise ValueError("Quaternion must have unit norm")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
