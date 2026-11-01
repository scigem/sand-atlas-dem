"""Frozen mesh-surface/SDF refinement check; exact zero-distance source oracle."""

import argparse
from pathlib import Path

import numpy as np

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = read_json(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    coarse = Path(policy["coarse_library"])
    fine = Path(policy["fine_library"])
    if (
        sha256(coarse) != policy["coarse_sha256"]
        or sha256(fine) != policy["fine_sha256"]
    ):
        raise ValueError("Library identity changed")
    a = ShapeLibrary.load(coarse)
    b = ShapeLibrary.load(fine)
    amap = {s.id: s for s in a.shapes}
    rows = []
    ae = []
    be = []
    if set(amap) != {s.id for s in b.shapes}:
        raise ValueError("Different grain identities")
    for shape in b.shapes:
        old = amap[shape.id]
        if (
            old.provenance["source_sha256"] != shape.provenance["source_sha256"]
            or not np.array_equal(old.vertices, shape.vertices)
            or old.volume != shape.volume
        ):
            raise ValueError("Physical geometry changed")
        diameter = (6 * shape.volume / np.pi) ** (1 / 3)
        va, ga = old.query(shape.samples)
        vb, gb = shape.query(shape.samples)
        xa = abs(va) / diameter
        xb = abs(vb) / diameter
        if not np.isfinite(xa).all() or not np.isfinite(xb).all():
            raise ValueError("Surface outside field padding")
        ae.extend(xa)
        be.extend(xb)
        rows.append(
            dict(
                shape=shape.id,
                source_sha256=shape.provenance["source_sha256"],
                coarse_p95_fraction_diameter=float(np.quantile(xa, 0.95)),
                fine_p95_fraction_diameter=float(np.quantile(xb, 0.95)),
                fine_maximum_error_m=float(abs(vb).max()),
                fine_gradient_norm_error_p95=float(
                    np.quantile(abs(np.linalg.norm(gb, axis=1) - 1), 0.95)
                ),
            )
        )
    coarse95 = float(np.quantile(ae, 0.95))
    fine95 = float(np.quantile(be, 0.95))
    worst = max(r["fine_p95_fraction_diameter"] for r in rows)
    ratio = coarse95 / fine95
    limits = policy["limits"]
    passed = (
        fine95 <= limits["pooled_fine_p95_fraction_diameter"]
        and worst <= limits["worst_grain_fine_p95_fraction_diameter"]
        and ratio >= limits["minimum_pooled_p95_improvement"]
    )
    report = dict(
        protocol=policy,
        protocol_sha256=sha256(args.protocol),
        script_sha256=sha256(__file__),
        coarse_pooled_p95=coarse95,
        fine_pooled_p95=fine95,
        worst_grain_fine_p95=worst,
        improvement_ratio=ratio,
        rows=rows,
        passed=passed,
    )
    write_json(args.output / "report.json", report)
    print(passed, coarse95, fine95, worst, ratio)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
